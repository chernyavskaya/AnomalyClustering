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
from data_utils.data_processing import GraphDataset, DenseEventDataset, GraphDatasetOnline
from training_utils.metrics import cluster_acc
from models.models import DenseAE, GraphAE, IDEC 
from training_utils.activation_funcs  import get_activation_func
from training_utils.training import pretrain_ae_dense,train_test_ae_dense,train_test_idec_dense,pretrain_ae_graph, train_test_ae_graph,train_test_idec_graph, target_distribution, save_ckp, create_ckp, load_ckp,export_jsondump

from training_utils.plot_losses import loss_curves

import os.path as osp
sys.path.append(os.path.abspath(os.path.join('../../')))
sys.path.append(os.path.abspath(os.path.join('../')))
import training_utils.model_summary as summary

torch.autograd.set_detect_anomaly(True)
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
                        input_shape_global = 3)
    if args.load_ae!='' :
        if osp.isfile(args.load_ae):
            pretrain_path = args.load_ae
        else :
            print('Requested pretrained model is not found. Exiting.'.format(args.load_ae))
            exit()
    else :
        pretrain_path = output_path+'/pretrained_AE.pkl'


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
    if args.load_ae!='' :
        model.ae, optimizer_ae, scheduler_ae, start_epoch, _,_ = load_ckp(pretrain_path, model.ae, optimizer_ae, scheduler_ae)
        model.ae.to(device)
    summary_writer = SummaryWriter(log_dir=osp.join(output_path,'tensorboard_logs_ae/'))
    pretrain_ae_graph(model.ae,train_loader,test_loader,optimizer_ae,start_epoch,start_epoch+args.n_epochs,pretrain_path,device,scheduler_ae,summary_writer,pid_weight,pid_loss_weight,met_loss_weight)
    summary_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--load_ae', type=str, default='')
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--n_clusters', default=6, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--latent_dim', default=8, type=int)
    parser.add_argument('--input_shape', default='18,5', type=str)
    parser.add_argument('--num_pid_classes', default=4, type=int)
    parser.add_argument('--hidden_channels', default='30,30,30', type=str)  
    parser.add_argument('--dropout', default=0.01, type=float)  
    parser.add_argument('--activation', default='leakyrelu_0.1', type=str)  
    parser.add_argument('--n_run', type=int) 
    parser.add_argument('--generator', default=1, type=int)
    parser.add_argument('--n_epochs',default=100, type=int)

    args = parser.parse_args()
    if args.n_run is None:
        print('Please set run number to save the output. Exiting.')
        exit()
    args.hidden_channels = [int(s) for s in args.hidden_channels.replace(' ','').split(',')]
    args.input_shape = [int(s) for s in args.input_shape.replace(' ','').split(',')]
    base_output_path = '/eos/user/n/nchernya/MLHEP/AnomalyDetection/autoencoder_for_anomaly/graph_AD_event_based/'
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

    DATA_PATH = '/eos/user/n/nchernya/MLHEP/AnomalyDetection/autoencoder_for_anomaly/files/AD_event_based/graph_data/'
    BG_NAME = 'background_train_fixed.h5'

    filename = DATA_PATH + BG_NAME 
    if not args.generator:
        in_file = h5py.File(filename, 'r') 
        file_dataset = np.array(in_file['Particles'])
        truth_dataset = np.array(in_file['ProcessID'])
        file_dataset =  data_proc.prepare_ad_event_based_dataset(file_dataset,truth_dataset=truth_dataset,tot_evt=1e4,shuffle=True)
        prepared_dataset,datas =  data_proc.prepare_graph_datas(file_dataset,args.input_shape[0],n_top_proc = -1,connect_only_real=True)

        train_test_split = 0.8
        train_len = int(len(datas)*train_test_split)
        test_len = len(datas)-train_len
        train_dataset = GraphDataset(datas[0:train_len])
        test_dataset = GraphDataset(datas[train_len:])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True) #,num_workers=5
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True) #,num_workers=5
    else:

        train_dataset = GraphDatasetOnline(root=DATA_PATH,input_files=[BG_NAME],datasetname='Particles',truth_datasetname='ProcessID',
                                  n_events=1e6,data_chunk_size=int(2e4),
                                  input_shape=[18,5],connect_only_real=True, 
                                  shuffle=True)
                        
        test_dataset = GraphDatasetOnline(root=DATA_PATH,input_files=[BG_NAME.replace('train','test')],datasetname='Particles',truth_datasetname='ProcessID',
                                  n_events=1e5,data_chunk_size=int(2e4),
                                  input_shape=[18,5],connect_only_real=True, 
                                  shuffle=True)


        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True) #,num_workers=5
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True) #,num_workers=5

    pid_weight = data_proc.get_relative_weights(np.array(h5py.File(filename, 'r')['Particles'])[:,1:,-1].reshape(-1),mode='max')
    print(pid_weight,type(pid_weight))
    #pid_weight_file = open(output_path+"/pid_weight.txt", "w")
    #for row in pid_weight:
    #    np.savetxt(pid_weight_file, row)
    pid_weight = torch.tensor(pid_weight).float().to(device)


    #this should be parametrizable 
    pid_loss_weight = 0.5 #0.1
    met_loss_weight = 5.0 #5

    print(args)
    train_idec()
