# Developed by © chernyavskaya 
#
#
from __future__ import print_function, division
import setGPU
import os,sys
import argparse
import numpy as np
import random
import h5py, json, glob, tqdm, math, random

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader as DataLoaderTorch
from torch.utils.data.dataset import random_split

from torch_geometric.data import Data, Batch, DataLoader

import data_utils.data_processing as data_proc
from data_utils.data_processing import GraphDataset, DenseEventDataset
from training_utils.metrics import cluster_acc
from models.models import DenseAE, GraphAE, IDEC 
from training_utils.training import target_distribution,pretrain_ae_graph, train_test_ae_graph,train_test_idec_graph
from training_utils.activation_funcs  import get_activation_func

import os.path as osp
sys.path.append(os.path.abspath(os.path.join('../../')))
sys.path.append(os.path.abspath(os.path.join('../')))
import ADgvae.utils_torch.model_summary as summary

#torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_idec():

    model_AE = GraphAE(input_shape = args.input_shape,
                        hidden_channels = args.hidden_channels,
                        latent_dim = args.latent_dim,
                        activation=args.activation,
                        dropout=args.dropout,
                        input_shape_global = 2)
    model = IDEC(AE = model_AE,
                input_shape = args.input_shape, 
                hidden_channels = args.hidden_channels,
                latent_dim = args.latent_dim,
                n_clusters=args.n_clusters,
                alpha=1,
                device=device,
                pretrain_path=args.pretrain_path)
    model.to(device)
    summary.gnn_model_summary(model)

    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)

    if args.retrain_ae :
        pretrain_ae_graph(model.ae,train_loader,test_loader,optimizer,args.n_epochs,args.pretrain_path,device,pid_weight,pid_loss_weight,met_loss_weight,energy_loss_weight)
    else :
        model.ae.load_state_dict(torch.load(args.pretrain_path))
        model.ae.to(device)
        print('load pretrained ae from', args.pretrain_path)

    print('Initializing cluster center with pre-trained weights')
    kmeans_initialized = None
    if args.full_kmeans:
        print('Full k-means')
        #Full k-means if dataset can fit in memeory (better cluster initialization and faster convergence of the model)
        kmeans_loader = DataLoaderTorch(dataset, batch_size=len(dataset), shuffle=True,drop_last=True) #,num_workers=5
        kmeans_initialized = KMeans(n_clusters=args.n_clusters, n_init=20)
    else:
        kmeans_loader = train_loader
        #Mini batch k-means if k-means on full dataset cannot fit in memory, fast but slower convergence of the model as cluster initialization is not as good
        print('Minbatch k-means')
        mbk = MiniBatchKMeans(n_clusters=args.n_clusters, n_init=20, batch_size=args.batch_size)
    for i, data in enumerate(kmeans_loader):
        data = data.to(device)
        model.clustering(mbk,data,args.full_kmeans,kmeans_initialized)

    pred_labels_last = 0
    delta_label = 1e4
    model.train()
    for epoch in range(1):#:range(args.n_epochs):

        #evaluation part
        if epoch % args.update_interval == 0:

            p_all = []
            for i, data in enumerate(train_loader):
                data = data.to(device)

                _,_, tmp_q_i, _ = model(data)
                tmp_q_i = tmp_q_i.data
                p = target_distribution(tmp_q_i)
                p_all.append(p)

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
                torch.save(model.state_dict(), args.pretrain_path.replace('.pkl','_fullmodel_num_clust_{}.pkl'.format(args.n_clusters)))
                print("Full model saved")
                break

        #training part
        train_loss,train_kl_loss,train_reco_loss,train_pid_loss,train_energy_loss, train_met_loss = train_test_idec_graph(model,train_loader,p_all,optimizer,device,args.gamma,pid_weight,pid_loss_weight,met_loss_weight,energy_loss_weight,mode='train')
        test_loss,test_kl_loss,test_reco_loss,test_pid_loss,test_energy_loss, test_met_loss = train_test_idec_graph(model,test_loader,p_all,optimizer,device,args.gamma,pid_weight,pid_loss_weight,met_loss_weight,energy_loss_weight,mode='test')

        print("epoch {} : TRAIN : total loss={:.4f}, kl loss={:.4f}, reco loss={:.4f}, pid loss={:.4f}, energy loss={:.4f}, met loss={:.4f}".format(epoch, train_loss, train_kl_loss, train_reco_loss,train_pid_loss,train_energy_loss, train_met_loss   ))
        print("epoch {} : TEST : total loss={:.4f}, kl loss={:.4f}, reco loss={:.4f}, pid loss={:.4f}, energy loss={:.4f}, met loss={:.4f}".format(epoch, test_loss, test_kl_loss, test_reco_loss,test_pid_loss,test_energy_loss, test_met_loss ))
        if epoch==args.n_epochs-1 :
            torch.save(model.state_dict(), args.pretrain_path.replace('.pkl','_fullmodel_num_clust_{}.pkl'.format(args.n_clusters)))
        torch.save(model.state_dict(), args.pretrain_path.replace('.pkl','_fullmodel_num_clust_{}.pkl'.format(args.n_clusters)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--n_top_proc', type=int, default=3)
    parser.add_argument('--full_kmeans', type=int, default=0)
    parser.add_argument('--retrain_ae', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--n_clusters', default=5, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--latent_dim', default=5, type=int)
    parser.add_argument('--input_shape', default=[16,5], type=int)
    parser.add_argument('--hidden_channels', default=[8, 12, 16, 20, 25, 30, 40, 60,50,40,30 ], type=int)  
    parser.add_argument('--dropout', default=0.0, type=float)  
    parser.add_argument('--activation', default='leakyrelu_0.5', type=str)  
    parser.add_argument('--pretrain_path', type=str, default='data_graph/graph_ae_pretrain.pkl') 
    parser.add_argument('--gamma',default=100.,type=float,help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--n_epochs',default=100, type=int)
    args = parser.parse_args()
    args.activation = get_activation_func(args.activation)

    DATA_PATH = '/eos/user/n/nchernya/MLHEP/AnomalyDetection/AnomalyClustering/inputs/'
    TRAIN_NAME = 'background_chan3_passed_ae_l1.h5'
    filename_bg = DATA_PATH + TRAIN_NAME 
    in_file = h5py.File(filename_bg, 'r') 
    file_dataset = np.array(in_file['dataset'])
    #trying temp to see what happens if we separate peak of 0s from eta and phi (activation function and pi cyclicity was changed in the model accordingly)
    file_dataset[:,1:,4] = np.where(file_dataset[:,1:,1]==0.,0.,file_dataset[:,1:,4]+3.0)
    file_dataset[:,1:,5] = np.where(file_dataset[:,1:,1]==0.,0.,file_dataset[:,1:,5]+3.4)

    prepared_dataset,datas =  data_proc.prepare_graph_datas(file_dataset,args.input_shape[0],n_top_proc = args.n_top_proc,connect_only_real=True)

    pid_weight = data_proc.get_relative_weights(prepared_dataset[:,1:,1].reshape(prepared_dataset[:,1:,1].shape[0]*prepared_dataset[:,1:,1].shape[1]),mode='max')
    #pid_weight = [1.,1.4,5.,5.]
    pid_weight = torch.tensor(pid_weight).float().to(device)

    train_test_split = 0.99
    train_len = int(len(datas)*train_test_split)
    test_len = len(datas)-train_len
    random.shuffle(datas)
    train_dataset = GraphDataset(datas[0:train_len])
    test_dataset = GraphDataset(datas[train_len:])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True) #,num_workers=5
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True) #,num_workers=5

    #this should be parametrizable 
    energy_loss_weight = 1.0
    pid_loss_weight = 0.1
    met_loss_weight = 10.0

    print(args)
    train_idec()
