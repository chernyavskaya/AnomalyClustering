# -*- coding: utf-8 -*-
#
# Starting code for iDEC from © dawnranger
# The rest of development from © chernyavskaya 
#
#
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
from torch.nn import Embedding
from torch.autograd import Variable

from scipy.sparse import csr_matrix
from torch_geometric.nn import Sequential, GCN, GCNConv, EdgeConv, GATConv, GATv2Conv, global_mean_pool, DynamicEdgeConv, BatchNorm

from torch_geometric.utils import from_scipy_sparse_matrix, to_dense_batch

from torch_geometric.data import Data, Batch, DataLoader
from torch.utils.data import DataLoader as DataLoaderTorch

from utils import GraphDataset, cluster_acc, DenseEventDataset
from torch_scatter import scatter_mean,scatter_max

import os.path as osp
sys.path.append(os.path.abspath(os.path.join('../../')))
sys.path.append(os.path.abspath(os.path.join('../')))
import ADgvae.utils_torch.model_summary as summary

#torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
eps = 1e-12
PI = math.pi
TWOPI = 2*math.pi


def cycle_by_2pi(in_tensor):
    in_tensor  = torch.where(in_tensor >= PI, in_tensor-TWOPI, in_tensor)
    in_tensor  = torch.where(in_tensor < -PI, in_tensor+TWOPI, in_tensor)
    return in_tensor


def huber_mask(inputs, outputs):
    # we might want to introduce different weighting to the parts of the loss with  jets/muons/...
    loss_fnc = torch.nn.HuberLoss(delta=10.0)
    loss = loss_fnc(inputs,outputs)
    return loss



class DenseAE(torch.nn.Module):
  def __init__(self, input_shape, hidden_channels,latent_dim,activation=nn.LeakyReLU(negative_slope=0.5),input_shape_global = 2):
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
    for i in range(0,len(hidden_channels)-1):
        self.enc_convs.append(Linear(hidden_channels[i], hidden_channels[i+1]))
        self.enc_convs.append(nn.BatchNorm1d(hidden_channels[i+1]))
    self.num_enc_convs  = len(self.enc_convs)
    self.z_layer = Linear(hidden_channels[-1], latent_dim) 

    # DECODER

    self.dec_convs = ModuleList()
    self.dec_convs.append(Linear(latent_dim, hidden_channels[-1]))
    self.dec_convs.append(nn.BatchNorm1d(hidden_channels[-1]))
    for i in range(len(hidden_channels)-1,0,-1):
        self.dec_convs.append(Linear(hidden_channels[i], hidden_channels[i-1]))
        self.dec_convs.append(nn.BatchNorm1d(hidden_channels[i-1]))
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
        x_bar[:,self.access_idx[i_obj]+i_feat:self.access_idx[i_obj]+num_obj*self.num_feats:self.num_feats] = F.relu(dec[:,self.access_idx[i_obj]+i_feat:self.access_idx[i_obj]+num_obj*self.num_feats:self.num_feats])
    
    x_bar = input_zeros_mask * x_bar

    return x_bar,1, z


class IDEC(nn.Module):

    def __init__(self,
                input_shape, 
                hidden_channels,
                latent_dim,
                n_clusters,
                alpha=1,
                pretrain_path='data_dense/dense_ae_pretrain.pkl'):
        super(IDEC, self).__init__()
        self.alpha = 1.0
        self.pretrain_path = pretrain_path

        self.ae = DenseAE(
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

    def forward(self, data):
        x_bar, _, z = self.ae(data)
        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x_bar, _, q, z

    def clustering(self,mbk,data):
        self.eval()
        _,_,pred_labels_ae = self.ae(data)
        pred_labels_ae = pred_labels_ae.data.cpu().numpy()
        pred_labels = mbk.partial_fit(pred_labels_ae) #seems we can only get a centre from batch
        self.cluster_centers = mbk.cluster_centers_ #keep the cluster centers
        self.updateClusterCenter(self.cluster_centers)

    def updateClusterCenter(self, cc):
        self.cluster_layer.data = torch.from_numpy(cc).to(device)


    def validateOnCompleteTestData(self,test_loader):
        self.eval()
        pred_labels = np.array([self.forward(x.to(device))[2].data.cpu().numpy().argmax(1) for i,(x,_) in enumerate(test_loader)]) #argmax(1) #index (cluster nubmber) of the cluster with the highest probability q.
        latent_pred = np.array([self.forward(x.to(device))[3].data.cpu().numpy() for i,(x,_) in enumerate(test_loader)])
        true_labels = np.array([y.cpu().numpy() for i,(_,y) in enumerate(test_loader)])
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
    train_loader = DataLoaderTorch(dataset, batch_size=args.batch_size, shuffle=True,drop_last=True) #,num_workers=5
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.n_epochs):
        total_loss = 0.
        for i, (x,_) in enumerate(train_loader):
            x = x.to(device)

            optimizer.zero_grad()
            x_bar,_, z = model(x)

            loss = huber_mask(x,x_bar)

            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("epoch {} : total loss={:.4f}".format(epoch, total_loss / (i + 1) ))

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

    train_loader = DataLoaderTorch(
        dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    optimizer = Adam(model.parameters(), lr=args.lr)

    print('Initializing cluster center with pre-trained weights')
    if args.full_kmeans:
        print('Full k-means')
        #Full k-means if dataset can fit in memeory (better cluster initialization and faster convergence of the model)
        full_dataset_loader = DataLoaderTorch(dataset, batch_size=len(dataset), shuffle=True,drop_last=True) #,num_workers=5
        for i, (x,_) in enumerate(full_dataset_loader):
            x = x.to(device)
            _,_, hidden = model.ae(x)
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
            y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
            model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    else:
        #Mini batch k-means if k-means on full dataset cannot fit in memory, fast but slower convergence of the model as cluster initialization is not as good
        print('Minbatch k-means')
        mbk = MiniBatchKMeans(n_clusters=args.n_clusters, n_init=20, batch_size=args.batch_size)
        #step 1 - get cluster center from batch
        #here we are using minibatch kmeans to be able to cope with larger dataset.
        for i, (x,_) in enumerate(train_loader):
            x = x.to(device)
            model.clustering(mbk,x)

    pred_labels_last = 0
    delta_label = 1e4
    model.train()
    for epoch in range(1):#:range(args.n_epochs):

        #evaluation part
        if epoch % args.update_interval == 0:

            p_all = []
            for i, (x,y) in enumerate(train_loader):
                x = x.to(device)

                _,_, tmp_q_i, _ = model(x)
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
        total_loss,total_kl_loss  = 0.,0.
        for i, (x,_) in enumerate(train_loader):
            x = x.to(device)

            x_bar, _, q, _ = model(x)

            reco_loss = huber_mask(x,x_bar)
            kl_loss = F.kl_div(q.log(), p_all[i],reduction='batchmean')
            loss = args.gamma * kl_loss + reco_loss 

            optimizer.zero_grad()
            total_loss += loss.item()
            total_kl_loss += kl_loss.item()

            loss.backward()
            optimizer.step()
        print("epoch {} : total loss={:.4f}, kl loss={:.4f} ".format(epoch, total_loss / (i + 1), total_kl_loss / (i + 1) ))
        if epoch==args.n_epochs-1 :
            torch.save(model.state_dict(), args.pretrain_path.replace('.pkl','_fullmodel_num_clust_{}.pkl'.format(args.n_clusters)))
        torch.save(model.state_dict(), args.pretrain_path.replace('.pkl','_fullmodel_num_clust_{}.pkl'.format(args.n_clusters)))


def prepare_1d_datasets(file_dataset):
    proc_idx = 0
    dkl_idx = 1
    charge_idx = 2
    features_idx = [3,4,5,6]

    dataset_proc_truth = file_dataset[:,0,[proc_idx]]
    dataset_2d = file_dataset[:,:,features_idx]
    dataset_2d[:,:,0] = dataset_2d[:,:,0]/1e5 #E
    dataset_2d[:,:,1] = dataset_2d[:,:,1]/1e5 #pT
    #log of energy and pt as preprocessing
    dataset_2d[:,:,0] = np.log(dataset_2d[:,:,0]+1)
    dataset_2d[:,:,1] = np.log(dataset_2d[:,:,1]+1) 
    dataset_2d[:,0,1] = 0. #met pt is 0
    dataset_2d[:,0,2] = 0. #met eta is 0


    #dataset_met = dataset_2d[:,0,[0,-1]] #met is first , and only has E and phi
    #dataset_particles = dataset_2d[:,1:,:] #other particles follow , and have all features 
    #dataset_particles = dataset_particles.reshape((dataset_particles.shape[0],dataset_particles.shape[1]*dataset_particles.shape[2]))
    #dataset_1d = np.hstack([dataset_met,dataset_particles])
    
    dataset_1d = dataset_2d.reshape((dataset_2d.shape[0],dataset_2d.shape[1]*dataset_2d.shape[2]))

    return dataset_1d,dataset_proc_truth


if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--full_kmeans', type=int, default=0)
    parser.add_argument('--retrain_ae', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--n_clusters', default=5, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--latent_dim', default=5, type=int)
    parser.add_argument('--input_shape', default=[124], type=int)
    parser.add_argument('--hidden_channels', default=[70,50,30,10 ], type=int)   
    parser.add_argument('--pretrain_path', type=str, default='data_dense/dense_ae_pretrain.pkl') 
    parser.add_argument('--gamma',default=100.,type=float,help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--n_epochs',default=100, type=int)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    #device='cpu'

    #try to add embedding layer for particle type
    #loss function for particle type should be different if no embedding is used (cross entropy)
    #think about normalization and whether to separate 0
    #go back to trying activation function 

    DATA_PATH = '/eos/user/n/nchernya/MLHEP/AnomalyDetection/AnomalyClustering/inputs/'
    TRAIN_NAME = 'bkg_sig_0.0156_l1_filtered_padded.h5'
    filename_bg = DATA_PATH + TRAIN_NAME 
    in_file = h5py.File(filename_bg, 'r') 
    file_dataset = np.array(in_file['dataset'])
    file_dataset_1d,file_dataset_proc_truth = prepare_1d_datasets(file_dataset)

    dataset = DenseEventDataset(file_dataset_1d,file_dataset_proc_truth)

    print(args)
    train_idec()
