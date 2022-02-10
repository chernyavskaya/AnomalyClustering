# -*- coding: utf-8 -*-
#
# 
# Code developed by © chernyavskaya 
# Starting point for iDEC from © dawnranger
#
from __future__ import print_function, division
import os,sys
import argparse
import numpy as np
import h5py, json, glob, tqdm, math, random

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from torch.nn import ModuleList
from torch.nn import Embedding
from torch.autograd import Variable

from torch_geometric.nn import Sequential, GCN, GCNConv, EdgeConv, GATConv, GATv2Conv, global_mean_pool, DynamicEdgeConv, BatchNorm
from torch_geometric.utils import from_scipy_sparse_matrix, to_dense_batch
from torch_scatter import scatter_mean,scatter_max

from training_utils.metrics import cluster_acc
from training_utils.losses import cycle_by_2pi
from training_utils.activation_funcs  import get_activation_func
from training_utils.training import load_ckp

from models.layers import EdgeConvLayer, EmbeddingLayer
#torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
eps = 1e-12
PI = math.pi
TWOPI = 2*math.pi

def load_GraphAE(dictionary, device, checkpoint_path=None ):
    activation = get_activation_func(dictionary['activation'])
    model = GraphAE(input_shape = dictionary['input_shape'],
                    hidden_channels = dictionary['hidden_channels'],
                    latent_dim = dictionary['latent_dim'],
                    activation=activation,
                    dropout=dictionary['dropout'],
                    num_pid_classes=dictionary['num_pid_classes'],
                    input_shape_global = 2)
    if checkpoint_path!=None:
        model, _, _, _, _,_ = load_ckp(checkpoint_path, model, None, None)
    model.to(device)
    return model

def load_DenseAE(dictionary, device, checkpoint_path=None ):
    activation = get_activation_func(dictionary['activation'])
    model = DenseAE(input_shape = dictionary['input_shape'],
                    hidden_channels = dictionary['hidden_channels'],
                    latent_dim = dictionary['latent_dim'],
                    activation=activation,
                    dropout=dictionary['dropout'],)
    if checkpoint_path!=None:
        model, _, _, _, _,_ = load_ckp(checkpoint_path, model, None, None)
    model.to(device)
    return model

def load_IDEC(model_AE,dictionary, device, checkpoint_path=None ):
    model = IDEC(AE = model_AE,
                input_shape = dictionary['input_shape'],
                hidden_channels = dictionary['hidden_channels'],
                latent_dim = dictionary['latent_dim'],
                n_clusters=dictionary['n_clusters'],
                alpha=1,
                device=device)
    if checkpoint_path!=None:
        model, _, _, _, _,_ = load_ckp(checkpoint_path, model, None, None)
    model.to(device)
    return model


class GraphAE(torch.nn.Module):
  def __init__(self, input_shape, hidden_channels,latent_dim,activation=nn.LeakyReLU(negative_slope=0.1),dropout=0.05,input_shape_global = 2,num_pid_classes=4):
    super(GraphAE, self).__init__()

    #Which main block to use for the architecture
    layer = EdgeConvLayer # EdgeConvLayer #GCNConv #GCN
    layer_kwargs = {'act':activation, 'dropout':dropout}
    if layer == GCN :
        layer_kwargs['num_layers'] = 1

    self.activation = activation 
    #self.batchnorm = nn.BatchNorm1d(input_shape[-1])
    self.num_fixed_nodes = input_shape[0]
    self.input_shape_global = input_shape_global
    self.hidden_global = 4*input_shape_global
    self.num_feats = input_shape[1]
    self.num_pid_classes = num_pid_classes #4
    self.idx_cat = [0] #only pid for now
    self.idx_cont =  np.delete(np.arange(self.num_feats), self.idx_cat)
    self.energy_idx, self.pt_idx = 1,2
    self.eta_idx, self.phi_idx = 3,4

    #self.emb_szs = [[5,2]]  #list of lists of embeddings (pid : 5->2, charge : 3->2)
    #self.num_emb_feats = self.num_feats if len(self.idx_cat)==0 else self.num_feats - len(self.idx_cat) + sum(emb[-1] for emb in self.emb_szs)
    ##EMBEDDING of categorical variables (pid, charge)
    #self.embeddings = EmbeddingLayer(self.emb_szs)

    # ENCODER 
    self.enc_convs = ModuleList()  
    #self.enc_convs.append(layer(self.num_emb_feats, hidden_channels[0],activation=self.activation)) 
    self.enc_convs.append(layer(self.num_feats, hidden_channels[0],**layer_kwargs))
    for i in range(0,len(hidden_channels)-1):
        self.enc_convs.append(layer(hidden_channels[i], hidden_channels[i+1],**layer_kwargs))
    self.num_enc_convs  = len(self.enc_convs)
    #scatter_mean from batch_size x num_fixed_nodes -> batch_size
    self.enc_fc1 = torch.nn.Linear(2*hidden_channels[-1], hidden_channels[-1]) 
    self.enc_met_fc1 = torch.nn.Linear(self.input_shape_global, self.hidden_global)  
    self.enc_fc2 = torch.nn.Linear(hidden_channels[-1]+self.hidden_global, latent_dim) 

    # DECODER
    self.dec_fc1 = torch.nn.Linear(latent_dim, 2*latent_dim+self.hidden_global)
    self.dec_met_fc1 = torch.nn.Linear(self.hidden_global, self.input_shape_global)
    self.dec_fc2 = torch.nn.Linear(2*latent_dim, 2*latent_dim*self.num_fixed_nodes) #to map from tiled (batch_size x num_fixed_nodes) x latent space to a more meaningful weights between the nodes 
    #upsample from batch_size -> batch_size x num_fixed_nodes 
    #self.dec_fc1 = torch.nn.Linear(latent_dim, latent_dim) #to map from tiled (batch_size x num_fixed_nodes) x latent space to a more meaningful weights between the nodes 
    #self.dec_fc2 = torch.nn.Linear(latent_dim, 2*latent_dim)

    #reshape 
    self.dec_convs = ModuleList()
    self.dec_convs.append(layer(2*latent_dim, hidden_channels[-1],**layer_kwargs))
    for i in range(len(hidden_channels)-1,0,-1):
        self.dec_convs.append(layer(hidden_channels[i], hidden_channels[i-1],**layer_kwargs))
    #self.dec_convs.append(layer(hidden_channels[0], self.num_emb_feats ,activation=self.activation))
    layer_kwargs['act'] = nn.Identity()
    self.dec_convs.append(layer(hidden_channels[0], self.num_feats+self.num_pid_classes-1 ,**layer_kwargs))

    self.num_dec_convs  = len(self.dec_convs)


  def encode(self, data):
    x,x_met,edge_index, batch_index = data.x,data.x_met.reshape((-1,self.input_shape_global)), data.edge_index, data.batch
    #create learnable embedding of catgeorical variables
    #if len(self.idx_cat) > 0:
    #    x_cat = x[:,self.idx_cat]
    #    x_cont = x[:,self.idx_cont]
    #    x_cat = self.embeddings(x_cat.long())
    #    x = torch.cat([x_cat, x_cont], -1) 
    #self.embedded_input = x.clone()

    #Treat global graph features 
    x_met = self.enc_met_fc1(x_met)
    x_met = self.activation(x_met)

    # Obtain node embeddings 
    for i_layer in range(self.num_enc_convs):
        x = self.enc_convs[i_layer](x,edge_index)
    #reduce from all the nodes in the graph to 1 per graph
    x_mean = scatter_mean(x, batch_index, dim=0)
    x_max = scatter_max(x, batch_index, dim=0)[0]
    x = torch.cat((x_mean, x_max), dim=1)
    x = self.enc_fc1(x)
    x = self.activation(x)

    x_final = torch.cat((x_met,x),dim=-1)
    x_final = self.enc_fc2(x_final)
    #no activation function before latent space
    return x_final
 
  def decode(self, z, edge_index):
    encoded_x = self.dec_fc1(z)
    encoded_x = self.activation(encoded_x)

    x_met = encoded_x[:,0:self.hidden_global]
    x_met = self.dec_met_fc1(x_met)
    x_met_energy = F.relu(x_met[:,0:1]) #energy
    x_met_phi = x_met[:,1:]
    x_met_phi = cycle_by_2pi(x_met_phi)
    #x_met_phi = PI*torch.tanh(x_met_phi) #phi
    x_met = torch.cat([x_met_energy,x_met_phi], dim=-1) 

    x = encoded_x[:,self.hidden_global:]

    x = self.dec_fc2(x)
    x = self.activation(x)
    batch_size = x.shape[0]
    layer_size = x.shape[-1]
    #x = torch.reshape(x,(batch_size*self.num_fixed_nodes, int(layer_size/self.num_fixed_nodes)))
    ######
    ###try with permutation
    x = torch.reshape(x,(batch_size,self.num_fixed_nodes, int(layer_size/self.num_fixed_nodes)))
    idx = torch.randint(self.num_fixed_nodes, size=(batch_size, self.num_fixed_nodes)).to(x.get_device())
    x = x.gather(dim=1, index=idx.unsqueeze(-1).expand(x.shape))
    x = torch.reshape(x,(batch_size*self.num_fixed_nodes, int(layer_size/self.num_fixed_nodes)))
    ######
    for i_layer in range(self.num_dec_convs):
        x = self.dec_convs[i_layer](x,edge_index)
    x_cat = x[:,0:self.num_pid_classes]
    log_soft_max = nn.LogSoftmax(dim=-1)
    x_cat = log_soft_max(x_cat)

    x_eta = x[:,[self.num_pid_classes-1+self.eta_idx]]
    x_phi =  x[:,[self.num_pid_classes-1+self.phi_idx]]
    x_energy_pt = x[:,[self.num_pid_classes-1+self.energy_idx,self.num_pid_classes-1+self.pt_idx]]
    x_phi = cycle_by_2pi(x_phi)
    x_phi = PI*torch.tanh(x_phi)
    x_eta = 5.*torch.tanh(x_eta) # have a symmetric activation function that eventually dies out. 

#########
    #this scaling only applies when eta and pt are transformed to be > 0.
    #x_phi = 7.*torch.tanh(x_phi)
    #x_eta = 6.*torch.tanh(x_eta)
##########

    x_energy_pt = F.relu(x_energy_pt)
    #x_energy_pt = F.leaky_relu(x_energy_pt, negative_slope=0.1)
    #x = torch.cat([x_cat,x[:,self.num_pid_classes:]], dim=-1) 
    x_final = torch.cat([x_cat,x_energy_pt,x_eta,x_phi], dim=-1) 

    return x_final,x_met
 
  def forward(self,data):
    x,x_met,edge_index, batch_index = data.x,data.x_met.reshape((-1,self.input_shape_global)), data.edge_index, data.batch
    #x = self.batchnorm(x)
    z = self.encode(data)
    x_bar,x_met_bar = self.decode(z, edge_index)
    return x_bar,x_met_bar, z


class DenseAE(torch.nn.Module):
  def __init__(self, input_shape, hidden_channels,latent_dim,activation=nn.LeakyReLU(negative_slope=0.5),dropout=0.05):
    super(DenseAE, self).__init__()

    self.activation = activation 
    #self.batchnorm = nn.BatchNorm1d(input_shape[-1])
    self.input_shape = input_shape[0]
    self.num_feats = 4


    self.order_num_objets = [1,3,3,3,15,6]
    self.order_names_objects = 'met,e,g,mu,j,b'.split(',')
    self.feats = 'E,pt,eta,phi'.split(',')

    self.access_idx = [0] 
    for n_obj in self.order_num_objets:
        self.access_idx.append(self.access_idx[-1]+self.num_feats*n_obj)


    # ENCODER 
    self.enc_convs = ModuleList()  
    self.enc_convs.append(Linear(self.input_shape, hidden_channels[0]))
    self.enc_convs.append(nn.BatchNorm1d(hidden_channels[0]))
    self.enc_convs.append(nn.Dropout(p=dropout))
    for i in range(0,len(hidden_channels)-1):
        self.enc_convs.append(Linear(hidden_channels[i], hidden_channels[i+1]))
        self.enc_convs.append(nn.BatchNorm1d(hidden_channels[i+1]))
        self.enc_convs.append(nn.Dropout(p=dropout))
    self.num_enc_convs  = len(self.enc_convs)
    self.z_layer = Linear(hidden_channels[-1], latent_dim) 

    # DECODER

    self.dec_convs = ModuleList()
    self.dec_convs.append(Linear(latent_dim, hidden_channels[-1]))
    self.dec_convs.append(nn.BatchNorm1d(hidden_channels[-1]))
    self.dec_convs.append(nn.Dropout(p=dropout))
    for i in range(len(hidden_channels)-1,0,-1):
        self.dec_convs.append(Linear(hidden_channels[i], hidden_channels[i-1]))
        self.dec_convs.append(nn.BatchNorm1d(hidden_channels[i-1]))
        self.dec_convs.append(nn.Dropout(p=dropout))
    self.num_dec_convs  = len(self.dec_convs)
    self.x_bar_layer = Linear(hidden_channels[0], self.input_shape)

  def forward(self, input_x):
    #encoder
    input_zeros_mask = torch.ne(input_x,0).float().to(input_x.device)
    x = input_x
    for i_layer in range(self.num_enc_convs):
        x = self.enc_convs[i_layer](x)
        x = self.activation(x)
    z = self.z_layer(x) #no activation before latent space

    # decoder
    dec = z
    for i_layer in range(self.num_dec_convs):
        dec = self.dec_convs[i_layer](dec)
        dec = self.activation(dec)
    dec = self.x_bar_layer(dec)
    #activation depends on a feature
    #relu for E, pt, scaled tanh for phi and eta
    #should x_bar require gradient here ? 
    x_bar = torch.zeros(dec.shape[0],dec.shape[-1]).to(dec.device)
    i_eta = self.feats.index('eta')
    i_phi = self.feats.index('phi')
    i_pt = self.feats.index('pt')
    i_energy = self.feats.index('E')
    i_feat = i_eta
    for i_obj,num_obj in enumerate(self.order_num_objets):
        x_bar[:,self.access_idx[i_obj]+i_feat:self.access_idx[i_obj]+num_obj*self.num_feats:self.num_feats] = 4.*torch.tanh(dec[:,self.access_idx[i_obj]+i_feat:self.access_idx[i_obj]+num_obj*self.num_feats:self.num_feats])
    i_feat = i_phi
    for i_obj,num_obj in enumerate(self.order_num_objets):
        x_bar[:,self.access_idx[i_obj]+i_feat:self.access_idx[i_obj]+num_obj*self.num_feats:self.num_feats] = PI*torch.tanh(dec[:,self.access_idx[i_obj]+i_feat:self.access_idx[i_obj]+num_obj*self.num_feats:self.num_feats])
    for i_feat in [i_energy, i_pt]:
        for i_obj,num_obj in enumerate(self.order_num_objets):
            x_bar[:,self.access_idx[i_obj]+i_feat:self.access_idx[i_obj]+num_obj*self.num_feats:self.num_feats] = F.relu(dec[:,self.access_idx[i_obj]+i_feat:self.access_idx[i_obj]+num_obj*self.num_feats:self.num_feats])
    
    x_bar = input_zeros_mask * x_bar

    return x_bar,1, z


class IDEC(nn.Module):
    def __init__(self,
                AE,
                input_shape, 
                hidden_channels,
                latent_dim,
                n_clusters,
                device,
                alpha=1,
                pretrain_path='data_dense/dense_ae_pretrain.pkl'):
        super(IDEC, self).__init__()
        self.alpha = alpha
        self.device = device
        self.pretrain_path = pretrain_path

        self.ae = AE
        self.ae.to(self.device)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, latent_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)


    def forward(self, data):
        x_bar, x_global_bar, z = self.ae(data)
        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x_bar, x_global_bar, q, z

    def clustering(self,mbk,data,full_kmeans=False,kmeans_initialized=None):
        self.eval()
        _,_,pred_labels_ae = self.ae(data)

        if full_kmeans:
            y_pred = kmeans_initialized.fit_predict(pred_labels_ae.data.cpu().numpy())
            self.updateClusterCenter(kmeans_initialized.cluster_centers_)

        else:
            pred_labels = mbk.partial_fit(pred_labels_ae.data.cpu().numpy()) #seems we can only get a centre from batch
            self.cluster_centers = mbk.cluster_centers_ #keep the cluster centers
            self.updateClusterCenter(self.cluster_centers)

    def updateClusterCenter(self, cc):
        self.cluster_layer.data = torch.from_numpy(cc).to(self.device)


    def validateOnCompleteTestData(self,true_labels,pred_labels):
        self.eval()
        acc,reassignment = cluster_acc(true_labels, pred_labels)
        nmi = nmi_score(true_labels, pred_labels)
        ari = ari_score(true_labels, pred_labels)
        return acc, nmi, ari,reassignment


