# Developed by © chernyavskaya 
#
#
from __future__ import print_function, division
#import setGPU
import os,sys
import argparse
import pathlib
from pathlib import Path
import numpy as np
import random
import h5py, json, glob, tqdm, math, random

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from contextlib import redirect_stdout

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader as DataLoaderTorch
from torch.utils.data.dataset import random_split

from torch_geometric.data import Data, Batch, DataLoader

import data_utils.data_processing as data_proc
from data_utils.data_processing import GraphDataset, DenseEventDataset
from training_utils.metrics import cluster_acc
from models.models import DenseAE, GraphAE, IDEC 
from training_utils.activation_funcs  import get_activation_func
from training_utils.training import pretrain_ae_dense,train_test_ae_dense,train_test_idec_dense,pretrain_ae_graph, train_test_ae_graph,train_test_idec_graph, target_distribution, save_ckp, create_ckp, load_ckp,export_jsondump

from training_utils.plot_losses import loss_curves

import os.path as osp
sys.path.append(os.path.abspath(os.path.join('../../')))
sys.path.append(os.path.abspath(os.path.join('../')))
import training_utils.model_summary as summary

#torch.autograd.set_detect_anomaly(True)
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


def train_idec():

    model_AE = GraphAE(input_shape = args.input_shape,
                        hidden_channels = args.hidden_channels,
                        latent_dim = args.latent_dim,
                        activation=args.activation,
                        dropout=args.dropout,
                        num_pid_classes=args.num_pid_classes,
                        input_shape_global = 2)
    if args.load_ae!='' :
        if osp.isfile(args.load_ae):
            pretrain_path = args.load_ae
        else :
            print('Requested pretrained model is not found. Exiting.'.format(args.load_ae))
            exit()
    else :
        pretrain_path = output_path+'/pretrained_AE.pkl'

    if args.load_idec!='':
        if osp.isfile(args.load_idec):
            idec_path = args.load_idec
        else :
            print('Requested idec model is not found. Exiting.'.format(args.load_idec))
            exit()
    else :
        idec_path = output_path+'/idec_model.pkl'

    model = IDEC(AE = model_AE,
                input_shape = args.input_shape, 
                hidden_channels = args.hidden_channels,
                latent_dim = args.latent_dim,
                n_clusters=args.n_clusters,
                alpha=1,
                device=device)

    print(model)
    summary.gnn_model_summary(model)
    with open(os.path.join(output_path,'model_summary.txt'), 'w') as f:
        with redirect_stdout(f):
            print(model)
            print(summary.gnn_model_summary(model))


    start_epoch=0
    model.ae.to(device)#model has to be on device before constructing optimizers
    optimizer_ae = Adam(model.ae.parameters(), lr=args.lr)
    scheduler_ae = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ae, patience=3, threshold=10e-8,verbose=True,factor=0.1)
    if args.retrain_ae:
        if args.load_ae!='' :
            model.ae, optimizer_ae, scheduler_ae, start_epoch, _,_ = load_ckp(pretrain_path, model.ae, optimizer_ae, scheduler_ae)
            model.ae.to(device)
        summary_writer = SummaryWriter(log_dir=osp.join(output_path,'tensorboard_logs_ae/'))
        pretrain_ae_graph(model.ae,train_loader,test_loader,optimizer_ae,start_epoch,start_epoch+args.n_epochs,pretrain_path,device,scheduler_ae,summary_writer,pid_weight,pid_loss_weight,met_loss_weight)
        summary_writer.close()
    else :
        model.ae, optimizer_ae, scheduler_ae, start_epoch,_,_ = load_ckp(pretrain_path, model.ae, optimizer_ae, scheduler_ae)
        model.ae.to(device)
        print('load pretrained ae from', pretrain_path)


    start_epoch=0
    model.to(device)
    optimizer = Adam(model.ae.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, threshold=10e-8,verbose=True,factor=0.5)
    summary_writer = SummaryWriter(log_dir=osp.join(output_path,'tensorboard_logs_idec/'))
    if args.load_idec!='' :
        model, optimizer, scheduler, start_epoch, _,_ = load_ckp(idec_path, model, optimizer, scheduler)
    model.to(device)


    #check if we save center parameters in the checkpoitn, so we can restart the training of idec from the previous centers.
    print('Initializing cluster center with pre-trained weights')
    kmeans_initialized = KMeans(n_clusters=args.n_clusters, n_init=20)
    model.clustering(train_loader,kmeans_initialized)


    pred_labels_last = 0
    delta_label = 1e4
    best_test_loss=10000.
    for epoch in range(start_epoch,start_epoch+args.n_epochs_idec):

        #evaluation part
        model.eval()
        if epoch % args.update_interval == 0:

            #p_all = torch.zeros((len(train_loader),args.batch_size,args.n_clusters),device=device)
            p_all = []
            #for i, data in enumerate(train_loader):
            #    data = data.to(device)

            #    _,_, tmp_q_i, _ = model(data)
            #    tmp_q_i = tmp_q_i.data
            #    #p_all[i] = target_distribution(tmp_q_i)
            #    p_all.append(target_distribution(tmp_q_i))


            pred_labels = np.array([model.forward(data.to(device))[2].data.cpu().numpy().argmax(1) for i,data in enumerate(train_loader)]) #argmax(1) ##index (cluster nubmber) of the cluster with the highest probability q.
            true_labels = np.array([data.y.cpu().numpy() for i,data in enumerate(train_loader)])
            #reshape 
            pred_labels = np.reshape(pred_labels,pred_labels.shape[0]*pred_labels.shape[1])
            true_labels = np.reshape(true_labels,true_labels.shape[0]*true_labels.shape[1])


            #The evaluation of accuracy is done for information only, we cannot use the true labels for the training 
            acc, nmi, ari, _,  = model.validateOnCompleteTestData(true_labels,pred_labels)
            print('Iter {} :'.format(epoch),'Acc {:.4f}'.format(acc),
                  ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))

            #Caclulate by how much the prediction of labels changed w.r.t to previous evaluation, and stop the training if threshold is met
            if epoch==0 :
                pred_labels_last = pred_labels
            else:
                delta_label = np.sum(pred_labels != pred_labels_last).astype(
                np.float32) / pred_labels.shape[0]
                pred_labels_last = pred_labels
                print('delta ', delta_label)

            if epoch > 0 and delta_label < args.tol:
                print('delta_label {:.4f}'.format(delta_label), '< tol',
                      args.tol)
                print('Reached tolerance threshold. Stopping training.')
                checkpoint = create_ckp(epoch, train_loss,test_loss,model.state_dict(),optimizer.state_dict(), scheduler.state_dict())
                save_ckp(checkpoint, idec_path.replace('.pkl','_epoch_{}.pkl'.format(epoch+1)))
                print("Full model saved")
                break

        #training part
        model.train()

        train_loss,train_kl_loss,train_reco_loss,train_pid_loss, train_met_loss = train_test_idec_graph(model,train_loader,p_all,optimizer,device,args.gamma,pid_weight,pid_loss_weight,met_loss_weight,mode='train')
        test_loss,test_kl_loss,test_reco_loss,test_pid_loss, test_met_loss = train_test_idec_graph(model,test_loader,p_all,optimizer,device,args.gamma,pid_weight,pid_loss_weight,met_loss_weight,mode='test')


        print("epoch {} : TRAIN : total loss={:.4f}, kl loss={:.4f}, reco loss={:.4f}, pid loss={:.4f}, met loss={:.4f}".format(epoch, train_loss, train_kl_loss, train_reco_loss,train_pid_loss, train_met_loss   ))
        print("epoch {} : TEST : total loss={:.4f}, kl loss={:.4f}, reco loss={:.4f}, pid loss={:.4f}, met loss={:.4f}".format(epoch, test_loss, test_kl_loss, test_reco_loss,test_pid_loss, test_met_loss ))

        scheduler.step(train_loss)

        loss_names=["Loss Tot","Loss KL","Loss Reco","Loss Pid","Loss Energy","Loss Met"]
        for name, loss in zip(loss_names,[train_loss,train_kl_loss,train_reco_loss,train_pid_loss, train_met_loss]):
            summary_writer.add_scalar("Training "+ name, loss, epoch)
        for name, loss in zip(loss_names,[test_loss,test_kl_loss,test_reco_loss,test_pid_loss, test_met_loss]):
            summary_writer.add_scalar("Validation "+ name, loss, epoch)   
        for layer_name, weight in model.named_parameters():
            summary_writer.add_histogram(layer_name,weight, epoch)
            if layer_name!='cluster_layer':
                summary_writer.add_histogram(f'{layer_name}.grad',weight.grad, epoch)

        if epoch>=10 and epoch%10==0:
            checkpoint = create_ckp(epoch, train_loss,test_loss,model.state_dict(),optimizer.state_dict(), scheduler.state_dict())
            save_ckp(checkpoint, idec_path.replace('.pkl','_epoch_{}.pkl'.format(epoch+1)))
            print('New checkpoint for epoch {} saved'.format(epoch+1))

        if epoch>=10 and test_loss < best_test_loss*1.01: #allow variation within 1%
            best_test_loss = test_loss
            checkpoint = create_ckp(epoch, train_loss,test_loss,model.state_dict(),optimizer.state_dict(), scheduler.state_dict())
            print('New best model saved')
            best_fpath = idec_path.replace(idec_path.rsplit('/', 1)[-1],'')+'best_model_IDEC.pkl'
            save_ckp(checkpoint, best_fpath)

        
    checkpoint = create_ckp(epoch, train_loss,test_loss,model.state_dict(),optimizer.state_dict(), scheduler.state_dict())
    save_ckp(checkpoint, idec_path.replace('.pkl','_epoch_{}.pkl'.format(epoch+1)))

    merged_data = export_jsondump(summary_writer)
    loss_curves(merged_data, osp.join(output_path,'fig_dir/idec/'))
    summary_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--n_top_proc_bg', type=int, default=-1)
    parser.add_argument('--signals', type=str, default='Ato4l,hChToTauNu,hToTauTau,LQ')
    parser.add_argument('--frac_sig', type=float,default=0.01)
    parser.add_argument('--retrain_ae', type=int, default=1)
    parser.add_argument('--load_ae', type=str, default='')
    parser.add_argument('--load_idec', type=str, default='')
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--n_clusters', default=6, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--latent_dim', default=10, type=int)
    parser.add_argument('--input_shape', default='5,4', type=str)
    parser.add_argument('--num_pid_classes', default=5, type=int)
    parser.add_argument('--hidden_channels', default='30,30,30', type=str)  
    parser.add_argument('--dropout', default=0.01, type=float)  
    parser.add_argument('--activation', default='leakyrelu_0.1', type=str)  
    parser.add_argument('--n_run', type=int) 
    parser.add_argument('--gamma',default=1.,type=float,help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.0, type=float)
    parser.add_argument('--n_epochs',default=100, type=int)
    parser.add_argument('--n_epochs_idec',default=1, type=int)

    args = parser.parse_args()
    if args.n_run is None:
        print('Please set run number to save the output. Exiting.')
        exit()
    args.hidden_channels = [int(s) for s in args.hidden_channels.replace(' ','').split(',')]
    args.input_shape = [int(s) for s in args.input_shape.replace(' ','').split(',')]
    args.signals = [s for s in args.signals.replace(' ','').split(',')]
    base_output_path = '/eos/user/n/nchernya/MLHEP/AnomalyDetection/autoencoder_for_anomaly/clustering/trained_output/graph/'
    output_path = base_output_path+'run_{}/'.format(args.n_run)
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_path+'/fig_dir/').mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_path+'/fig_dir/idec/').mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_path+'/fig_dir/ae/').mkdir(parents=True, exist_ok=True)
    args_dict = vars(args)
    save_params_json = json.dumps(args_dict) 
    with open(os.path.join(output_path,'parameters.json'), 'w', encoding='utf-8') as f_json:
        json.dump(save_params_json, f_json, ensure_ascii=False, indent=4)
        
    args.activation = get_activation_func(args.activation)

    DATA_PATH = '/eos/user/n/nchernya/MLHEP/AnomalyDetection/autoencoder_for_anomaly/clustering/inputs/clustering_inputs/'
    BG_NAMES = ['background']

    file_datasets = []
    tot_bg = 0
    for NAME in BG_NAMES+args.signals:
        if 'background' in NAME:
            NAME_file = NAME+'_with_kl.h5'   
        else:
            NAME_file = NAME+'_13TeV_PU50_with_kl.h5'

        filename = DATA_PATH + NAME_file 
        in_file = h5py.File(filename, 'r') 
        file_dataset = np.array(in_file['Particles_HLT'])
        file_dataset = file_dataset[:,:,[0,1,2,4]] #removing charge for now
        if 'background' in NAME:
            truth_dataset = np.array(in_file['PID'])
        else:
            truth_dataset =  np.ones(file_dataset[:,:,0].shape[0])
            truth_dataset.fill(data_proc.inverse_dict_map(data_proc.process_name_dict)[NAME])
        particle_id = file_dataset[:,:,[3]]
        for p_id in [2,3,4,5]:
            np.place(particle_id, particle_id==p_id, p_id-1)

        #have to concatenate truth dataset to the file_dataset
        file_dataset = np.concatenate([np.repeat(np.expand_dims(np.expand_dims(truth_dataset,axis=-1),axis=-1),
                                    file_dataset.shape[1],axis=1),
                                    particle_id, #pid
                                    file_dataset[:,:,0:3]], #pt,eta,phi
                                    axis=-1)

        random.shuffle(file_dataset)
        if 'background' in NAME:
            top_proc_mask =  data_proc.select_top_n_procs(file_dataset[:,0,0],n_top_proc=args.n_top_proc_bg)
            file_dataset = file_dataset[top_proc_mask]
            tot_bg = file_dataset.shape[0]
        else:
            tot_sig = file_dataset.shape[0]
            sig_events = int(args.frac_sig*tot_bg)
            file_dataset = file_dataset[:sig_events if sig_events<tot_sig else tot_sig]
        file_datasets.append(file_dataset)
    file_dataset = np.concatenate(file_datasets,axis=0)

    prepared_dataset,datas =  data_proc.prepare_graph_datas(file_dataset,args.input_shape[0],n_top_proc = -1,connect_only_real=True)

    pid_weight = data_proc.get_relative_weights(prepared_dataset[:,1:,1].reshape(-1),mode='max')
    pid_weight = torch.tensor(pid_weight).float().to(device)

    train_test_split = 0.9
    random.shuffle(datas)
    #datas = datas[:int(1e5)]
    train_len = int(len(datas)*train_test_split)
    test_len = len(datas)-train_len
    train_dataset = GraphDataset(datas[0:train_len])
    test_dataset = GraphDataset(datas[train_len:])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True) #,num_workers=5
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True) #,num_workers=5

    #this should be parametrizable 
    pid_loss_weight = 0.5 #0.1
    met_loss_weight = 5.0 #5

    print(args)
    train_idec()
