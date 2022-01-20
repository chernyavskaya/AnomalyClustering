# -*- coding: utf-8 -*-
#
# Copyright © dawnranger.
#
# 2018-05-08 10:15 <dawnranger123@gmail.com>
#
# Distributed under terms of the MIT license.
from __future__ import print_function, division
import setGPU
import os,sys
import argparse
import numpy as np
import h5py, json, glob, tqdm, math, random

from sklearn.cluster import MiniBatchKMeans, KMeans

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from torch.nn import ModuleList

from scipy.sparse import csr_matrix
from torch_geometric.nn import Sequential, GCNConv, EdgeConv, GATConv, GATv2Conv, global_mean_pool, DynamicEdgeConv, BatchNorm

from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data, Batch, DataLoader
from utils import GraphDataset, cluster_acc
from torch_scatter import scatter_mean,scatter_max

import os.path as osp
sys.path.append(os.path.abspath(os.path.join('../../ADgvae/')))
import utils_torch.model_summary as summary

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class EdgeConvLayer(nn.Module):   
    def __init__(self, in_dim, out_dim, dropout=0.1, batch_norm=True, activation=nn.ReLU(), aggr='mean'):
        super().__init__()
        self.activation = activation
        self.batch_norm = batch_norm
            
        if self.batch_norm:
            self.edgeconv = nn.Sequential(nn.Linear(2*(in_dim), out_dim),
                                   nn.BatchNorm1d(out_dim),
                                   activation,
                                   nn.Dropout(p=dropout)) 
        else :
            self.edgeconv = nn.Sequential(nn.Linear(2*(in_dim), out_dim),
                                   activation,
                                   nn.Dropout(p=dropout))             

        ###dropout in AE as a regularization 
        self.edgeconv = EdgeConv(nn=self.edgeconv,aggr=aggr)

    def forward(self, feature, edge_index):
        h = self.edgeconv(feature, edge_index)
    
        return h

class GraphAE(torch.nn.Module):
  def __init__(self, input_shape, hidden_channels,latent_dim):
    super(GraphAE, self).__init__()

    #Which main block to use for the architecture
    layer = EdgeConvLayer # EdgeConvLayer #GCNConv 
    # ENCODER 
    self.num_fixed_nodes = input_shape[0]
    self.num_feats = input_shape[1]
    self.enc_convs = ModuleList()
    self.enc_convs.append(layer(self.num_feats, hidden_channels[0]))
    for i in range(0,len(hidden_channels)-1):
        self.enc_convs.append(layer(hidden_channels[i], hidden_channels[i+1]))
    self.enc_convs.append(layer(hidden_channels[-1], latent_dim))
    self.num_enc_convs  = len(self.enc_convs)
    #scatter_mean from batch_size x num_fixed_nodes -> batch_size
    self.enc_fc1 = torch.nn.Linear(2*latent_dim, 2*latent_dim) 
    self.enc_fc2 = torch.nn.Linear(2*latent_dim, latent_dim) 

    # DECODER
    #upsample from batch_size -> batch_size x num_fixed_nodes 
    self.dec_fc1 = torch.nn.Linear(latent_dim, latent_dim) #to map from tiled (batch_size x num_fixed_nodes) x latent space to a more meaningful weights between the nodes 
    self.dec_fc2 = torch.nn.Linear(latent_dim, 2*latent_dim)
    self.dec_convs = ModuleList()
    self.dec_convs.append(layer(2*latent_dim, hidden_channels[-1]))
    for i in range(len(hidden_channels)-1,0,-1):
        self.dec_convs.append(layer(hidden_channels[i], hidden_channels[i-1]))
    self.dec_convs.append(layer(hidden_channels[0], self.num_feats))
    self.num_dec_convs  = len(self.dec_convs)


  def encode(self, x, edge_index,batch_index):
    # Obtain node embeddings 
    for i_layer in range(self.num_enc_convs):
        x = self.enc_convs[i_layer](x,edge_index)
        x = F.relu(x)
    #reduce from all the nodes in the graph to 1 per graph
    x_mean = scatter_mean(x, batch_index, dim=0)
    x_max = scatter_max(x, batch_index, dim=0)[0]
    x = torch.cat((x_mean, x_max), dim=1)
    x = self.enc_fc1(x)
    x = F.relu(x)
    x = self.enc_fc2(x)
    x = F.relu(x)
    return x
 
  def decode(self, z, edge_index):
    x = torch.repeat_interleave(z, self.num_fixed_nodes, dim=0)
    x = self.dec_fc1(x)
    x = torch.relu(x)
    x = self.dec_fc2(x)
    x = torch.relu(x)
    for i_layer in range(self.num_dec_convs):
        x = self.dec_convs[i_layer](x,edge_index)
        x = F.relu(x)
    return x
 
  def forward(self,x, edge_index,batch_index):
    #x, edge_index = data.x, data.edge_index # x, edge_index, batch = data.x, data.edge_index, data.batch
    z = self.encode(x, edge_index,batch_index)
    x_bar = self.decode(z, edge_index)
    return x_bar, z



class IDEC(nn.Module):

    def __init__(self,
                input_shape, 
                hidden_channels,
                latent_dim,
                n_clusters,
                alpha=1,
                pretrain_path='data_graph/graph_ae_pretrain.pkl'):
        super(IDEC, self).__init__()
        self.alpha = 1.0
        self.pretrain_path = pretrain_path

        self.ae = GraphAE(
                    input_shape=input_shape, 
                    hidden_channels=hidden_channels, 
                    latent_dim=latent_dim)
        self.ae.to(device)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, latent_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def pretrain(self, path=''):
        if path == '':
            pretrain_ae(self.ae)
        # load pretrain weights
        self.ae.load_state_dict(torch.load(self.pretrain_path))
        self.ae.to(device)
        print('load pretrained ae from', path)

    def forward(self, x,edge_index,batch_index):
        x_bar, z = self.ae(x,edge_index,batch_index)
        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x_bar, q, z

    def clustering(self,mbk,x,edge_index,batch_index):
        self.eval()
        _,pred_labels_ae = self.ae(x,edge_index,batch_index)
        pred_labels_ae = pred_labels_ae.data.cpu().numpy()
        pred_labels = mbk.partial_fit(pred_labels_ae) #seems we can only get a centre from batch
        self.cluster_centers = mbk.cluster_centers_ #keep the cluster centers
        self.updateClusterCenter(self.cluster_centers)

    def updateClusterCenter(self, cc):
        self.cluster_layer.data = torch.from_numpy(cc).to(device)


    def validateOnCompleteTestData(self,test_loader):
        self.eval()
        pred_labels = np.array([self.forward(d.x.to(device),d.edge_index.to(device),d.batch.to(device))[1].data.cpu().numpy().argmax(1) for i,d in enumerate(test_loader)]) #argmax(1) #index (cluster nubmber) of the cluster with the highest probability q.
        latent_pred = np.array([self.forward(d.x.to(device),d.edge_index.to(device),d.batch.to(device))[2].data.cpu().numpy() for i,d in enumerate(test_loader)])
        true_labels = np.array([d.y.cpu().numpy() for i,d in enumerate(test_loader)])
        #reshape
        pred_labels = np.reshape(pred_labels,pred_labels.shape[0]*pred_labels.shape[1])
        true_labels = np.reshape(true_labels,true_labels.shape[0]*true_labels.shape[1])
        latent_pred = np.reshape(latent_pred,(latent_pred.shape[0]*latent_pred.shape[1],latent_pred.shape[2]))

        acc,reassignment = cluster_acc(true_labels, pred_labels)
        nmi = nmi_score(true_labels, pred_labels)
        ari = ari_score(true_labels, pred_labels)
        return acc, nmi, ari,reassignment, true_labels, pred_labels, latent_pred

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def pretrain_ae(model):
    '''
    pretrain autoencoder
    '''
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,drop_last=True) #,num_workers=5
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.n_epochs):
        total_loss = 0.
        for i, data in enumerate(train_loader):
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            batch_index = data.batch.to(device)

            optimizer.zero_grad()
            x_bar, z = model(x,edge_index,batch_index)
            loss = F.mse_loss(x_bar, x)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("epoch {} loss={:.4f}".format(epoch,
                                            total_loss / (i + 1)))
        torch.save(model.state_dict(), args.pretrain_path)
    print("model saved to {}.".format(args.pretrain_path))


def train_idec():

    model = IDEC(input_shape = args.input_shape, 
                hidden_channels = args.hidden_channels,
                latent_dim = args.latent_dim,
                n_clusters=args.n_clusters,
                alpha=1,
                pretrain_path=args.pretrain_path)
    model.to(device)
    summary.gnn_model_summary(model)

    if args.retrain_ae :
        model.pretrain()
    else :
        model.pretrain(args.pretrain_path)

    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    optimizer = Adam(model.parameters(), lr=args.lr)

    print('Initializing cluster center with pre-trained weights')
    if args.full_kmeans:
        print('Full k-means')
        #Full k-means if dataset can fit in memeory (better cluster initialization and faster convergence of the model)
        full_dataset_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True,drop_last=True) #,num_workers=5
        for i, data in enumerate(full_dataset_loader):
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            batch_index = data.batch.to(device)
            _, hidden = model.ae(x,edge_index,batch_index)
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
            y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
            model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    else:
        #Mini batch k-means if k-means on full dataset cannot fit in memory, fast but slower convergence of the model as cluster initialization is not as good
        print('Minbatch k-means')
        mbk = MiniBatchKMeans(n_clusters=args.n_clusters, n_init=20, batch_size=args.batch_size)
        #step 1 - get cluster center from batch
        #here we are using minibatch kmeans to be able to cope with larger dataset.
        for i, data in enumerate(train_loader):
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            batch_index = data.batch.to(device)
            model.clustering(mbk,x,edge_index,batch_index)

    pred_labels_last = 0
    delta_label = 1e4
    model.train()
    for epoch in range(args.n_epochs):

        #evaluation part
        if epoch % args.update_interval == 0:

            p_all = []
            for i, data in enumerate(train_loader):
                x = data.x.to(device)
                edge_index = data.edge_index.to(device)
                batch_index = data.batch.to(device)

                _, tmp_q_i, _ = model(x,edge_index,batch_index)
                tmp_q_i = tmp_q_i.data
                p = target_distribution(tmp_q_i)
                p_all.append(p)

            acc, nmi, ari, _, _, pred_labels,_ = model.validateOnCompleteTestData(train_loader)
            if epoch==0 :
                pred_labels_last = pred_labels
            else:
                delta_label = np.sum(pred_labels != pred_labels_last).astype(
                np.float32) / pred_labels.shape[0]
                pred_labels_last = pred_labels
                print('delta ', delta_label)

            print('Iter {} :'.format(epoch),'Acc {:.4f}'.format(acc),
                  ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))

            if epoch > 0 and delta_label < args.tol:
                print('delta_label {:.4f}'.format(delta_label), '< tol',
                      args.tol)
                print('Reached tolerance threshold. Stopping training.')
                torch.save(model.state_dict(), args.pretrain_path.replace('.pkl','_fullmodel_num_clust_{}.pkl'.format(args.n_clusters)))
                print("Full model saved")
                break

        #training part
        for i, data in enumerate(train_loader):
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            batch_index = data.batch.to(device)

            x_bar, q, _ = model(x,edge_index,batch_index)

            reconstr_loss = F.mse_loss(x_bar, x)
            kl_loss = F.kl_div(q.log(), p_all[i])
            loss = args.gamma * kl_loss + reconstr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch==args.n_epochs-1 :
            torch.save(model.state_dict(), args.pretrain_path.replace('.pkl','_fullmodel_num_clust_{}.pkl'.format(args.n_clusters)))



if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--full_kmeans', type=int, default=0)
    parser.add_argument('--retrain_ae', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=5, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--latent_dim', default=10, type=int)
    parser.add_argument('--input_shape', default=[17,5], type=int)
    parser.add_argument('--hidden_channels', default=[4,4,3,3,2,2], type=int)
    parser.add_argument('--pretrain_path', type=str, default='data_graph/graph_ae_pretrain.pkl') #data/gcnae_test
    parser.add_argument('--gamma',default=0.1,type=float,help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--n_epochs',default=20, type=int)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    #device='cpu'

    DATA_PATH = '/eos/user/n/nchernya/MLHEP/AnomalyDetection/AnomalyClustering/inputs/'
    TRAIN_NAME = 'background_chan3_passed_ae_l1.h5'
    filename_bg = DATA_PATH + TRAIN_NAME 
    in_file = h5py.File(filename_bg, 'r') 
    file_dataset = np.array(in_file['dataset'])
    file_dataset[:,:,2] = file_dataset[:,:,2]/1e5
    file_dataset[:,:,3] = file_dataset[:,:,3]/1e5
    #Select top N processes only :
    n_proc = 3
    (unique, counts) = np.unique(file_dataset[:,:,0], return_counts=True)
    procs_sorted, counts_sorted = zip(*sorted(zip(unique, counts), key=lambda x: x[1],reverse=True))
    top_proc_mask = np.isin(file_dataset[:,0,0], procs_sorted[:n_proc]) #choose top 3
    file_dataset = file_dataset[top_proc_mask]

    datas = []
    tot_evt = int(1e4) # file_dataset.shape[0]# int(1e4)
    print('Preparing the dataset of {} events'.format(tot_evt))
    n_objs = 17
    adj = [csr_matrix(np.ones((n_objs,n_objs)) - np.eye(n_objs))]*tot_evt
    edge_index = [from_scipy_sparse_matrix(a)[0] for a in adj]      
    x = [torch.tensor(file_dataset[i_evt,:,1:], dtype=torch.float) for i_evt in range(tot_evt)]
    y = [torch.tensor(int(file_dataset[i_evt,0,0]), dtype=torch.int) for i_evt in range(tot_evt)]
    datas = [Data(x=x_jet, edge_index=edge_index_jet,y=torch.unsqueeze(u_jet, 0)) 
         for x_jet,edge_index_jet,u_jet in zip(x,edge_index,y)]
    print('Dataset of {} events prepared'.format(tot_evt))
    dataset  = GraphDataset(datas)

    print(args)
    train_idec()
