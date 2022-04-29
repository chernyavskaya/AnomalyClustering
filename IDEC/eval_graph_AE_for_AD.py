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
from data_utils.data_processing import GraphDataset, DenseEventDataset,GraphDatasetOnline
from training_utils.metrics import cluster_acc
from models.models import DenseAE, GraphAE, IDEC ,load_GraphAE
from training_utils.activation_funcs  import get_activation_func
from training_utils.training import pretrain_ae_dense,train_test_ae_dense,train_test_idec_dense,pretrain_ae_graph, train_test_ae_graph,train_test_idec_graph, target_distribution, save_ckp, create_ckp, load_ckp,export_jsondump, evaluate_ae_graph

from training_utils.plot_losses import loss_curves

import os.path as osp
sys.path.append(os.path.abspath(os.path.join('../../')))
sys.path.append(os.path.abspath(os.path.join('../')))
import training_utils.model_summary as summary

torch.autograd.set_detect_anomaly(True)
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--load_ae', type=str, default='')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--generator', default=1, type=int)
    parser.add_argument('--n_run', type=int) 

    args = parser.parse_args()
    if args.n_run is None:
        print('Please set run number to save the output. Exiting.')
        exit()

    output_path = os.path.dirname(args.load_ae)
    params_dict_path = output_path+'/parameters.json'
    params_dict = json.loads(json.load(open(params_dict_path)))
    model_AE = load_GraphAE(params_dict, device, checkpoint_path=args.load_ae )


    pid_weight = pd.read_json(json.load(open(output_path+'/pid_weight.json')),orient='values').values.reshape(-1)
    pid_weight = torch.tensor(pid_weight).float().to(device)

    output_path = output_path+'/evaluated/'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    DATA_PATH = '/eos/user/n/nchernya/MLHEP/AnomalyDetection/autoencoder_for_anomaly/files/AD_event_based/graph_data/validation/'
    #DATA_PATH = '/mnt/ceph/users/nadezda/AD_data/AD_event_based/validation/'

    BG_NAME = 'background_validation_fixed.h5'
    SIG_NAMES = 'Ato4l,hChToTauNu,hToTauTau,LQ'.split(',') 

    bg_dataset = GraphDatasetOnline(root=DATA_PATH,in_memory=(args.generator==0),
                                  input_files=[BG_NAME],
                                  datasetname='Particles',truth_datasetname='ProcessID',
                                  n_events=-1,n_events_per_file=-1,data_chunk_size=int(5e4),
                                  input_shape=[18,5],connect_only_real=True, 
                                  shuffle=True)
                        
    sig_dataset = GraphDatasetOnline(root=DATA_PATH,in_memory=(args.generator==0),
                                  input_files=['sig_'+s+'_fixed.h5' for s in SIG_NAMES],
                                  datasetname='Particles',truth_datasetname='ProcessID',
                                  n_events=-1,n_events_per_file=-1,data_chunk_size=int(5e4),
                                  input_shape=[18,5],connect_only_real=True, 
                                  shuffle=True)


    bg_loader = DataLoader(bg_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True,num_workers=5)
    sig_loader = DataLoader(sig_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True,num_workers=5)


    model_num = args.load_ae.split('epoch_')[1].replace('.pkl','.h5')
    out_bg_file = output_path+'/background_evaluated_model'+model_num
    out_sig_file = output_path+'/signals_evaluated_model'+model_num

    evaluate_ae_graph(model_AE,bg_loader,device,out_bg_file,pid_weight)
    evaluate_ae_graph(model_AE,sig_loader,device,out_sig_file,pid_weight)


                                      
                       