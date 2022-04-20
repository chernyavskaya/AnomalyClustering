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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--load_ae', type=str, default='')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--n_run', type=int) 

    args = parser.parse_args()
    if args.n_run is None:
        print('Please set run number to save the output. Exiting.')
        exit()

    output_path = os.path.dirname(args.load_ae)
    params_dict_path = output_path+'/parameters.json'
    params_dict = json.loads(json.load(open(params_dict_path)))
    model_AE = load_GraphAE(params_dict, device, checkpoint_path=args.load_ae )

    output_path = output_path+'/evaluated/'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    DATA_PATH = '/eos/user/n/nchernya/MLHEP/AnomalyDetection/autoencoder_for_anomaly/files/AD_event_based/graph_data/validation/'
    BG_NAME = 'background_validation.h5'

    filename = DATA_PATH + BG_NAME 
    in_file = h5py.File(filename, 'r') 
    file_dataset = np.array(in_file['Particles'])
    file_dataset =  data_proc.prepare_ad_event_based_dataset(file_dataset,1e3,shuffle=True)
    prepared_dataset,datas =  data_proc.prepare_graph_datas(file_dataset,params_dict['input_shape'][0],n_top_proc = -1,connect_only_real=True)
    #pid_weight = data_proc.get_relative_weights(prepared_dataset[:,1:,1].reshape(-1),mode='max')
    #pid_weight = torch.tensor(pid_weight).float().to(device)

    SIG_NAMES = 'Ato4l,hChToTauNu,hToTauTau,LQ'.split(',') 
    file_datasets_signals = []
    for SIG_NAME in SIG_NAMES:
        SIG_NAME_file = 'sig_'+SIG_NAME+'.h5'
        filename_sig = DATA_PATH + SIG_NAME_file 
        in_file_sig = h5py.File(filename_sig, 'r') 
        file_dataset_sig = np.array(in_file_sig['Particles'])
        file_dataset_sig =  data_proc.prepare_ad_event_based_dataset(file_dataset_sig,1e3,shuffle=True,truth_name=SIG_NAME)
        file_datasets_signals.append(file_dataset_sig)
    file_dataset_sig = np.concatenate(file_datasets_signals,axis=0)
    prepared_dataset_sig,datas_sig =  data_proc.prepare_graph_datas(file_dataset_sig,params_dict['input_shape'][0],n_top_proc = -1,connect_only_real=True)


    bg_dataset = GraphDataset(datas)
    bg_loader = DataLoader(bg_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True) #,num_workers=5
    sig_dataset = GraphDataset(datas_sig)
    sig_loader = DataLoader(sig_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True) #,num_workers=5


    pred_feats, pred_met, latent, loss_dict = evaluate_ae_graph(model_AE,bg_loader,device)
    pred_feats_sig, pred_met_sig, latent_sig, loss_dict_sig = evaluate_ae_graph(model_AE,sig_loader,device)


    num_classes = model_AE.num_pid_classes
    pred_feats_merged, pred_feats_per_batch, pred_met = data_proc.prepare_final_output_features(pred_feats,pred_met,num_classes,args.batch_size)
    pred_feats_merged_sig, pred_feats_per_batch_sig, pred_met_sig = data_proc.prepare_final_output_features(pred_feats_sig,pred_met_sig,num_classes,args.batch_size)

    input_feats_merged,input_feats_per_batch, input_feats_met,true_labels = data_proc.prepare_final_input_features(prepared_dataset,args.batch_size)
    input_feats_merged_sig,input_feats_per_batch_sig, input_feats_met_sig,true_labels_sig = data_proc.prepare_final_input_features(prepared_dataset_sig,args.batch_size)


    out_bg_file = output_path+'/background_evaluated.h5'
    out_sig_file = output_path+'/signals_evaluated.h5'
    data_proc.prepare_ad_event_based_h5file(out_bg_file,true_labels,input_feats_per_batch, input_feats_met,pred_feats_per_batch,pred_met,loss_dict)
    data_proc.prepare_ad_event_based_h5file(out_sig_file,true_labels_sig,input_feats_per_batch_sig, input_feats_met_sig,pred_feats_per_batch_sig,pred_met_sig,loss_dict_sig)


                                      
                       