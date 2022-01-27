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

from utils import GraphDataset, cluster_acc
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

class EmbeddingLayer(nn.Module):
    """
    Embedding layer.
    Automatically splits input Tensors based on embedding sizes;
    then, embeds each feature separately and concatenates the output
    back into a single outpuut Tensor.
    """

    def __init__(self, emb_szs):
        super().__init__()
        self.embeddings = nn.ModuleList([Embedding(in_sz, out_sz) for in_sz, out_sz in emb_szs])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = [emb(x[..., i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(x, dim=-1)
        return x


def make_adjacencies(particles):
    real_p_mask = particles[:,:,1] > 0. # construct mask for real particles
    adjacencies = (real_p_mask[:,:,np.newaxis] * real_p_mask[:,np.newaxis,:]).astype('float32')
    return adjacencies


def cycle_by_2pi(in_tensor):
    in_tensor  = torch.where(in_tensor >= PI, in_tensor-TWOPI, in_tensor)
    in_tensor  = torch.where(in_tensor < -PI, in_tensor+TWOPI, in_tensor)
    return in_tensor


def chamfer_loss(target, reco, batch):
    x = to_dense_batch(target, batch)[0]
    y = to_dense_batch(reco, batch)[0] 
    #dist = pairwise_distance(x,y)
    diff = pairwise_distance(x,y)

    #if coordinate and energy should be weighted differently in the total sum of distances, this should be done here!
    #should at least check the magnitide of it
    #diff_0 = diff[:,:,:,0]
    #diff_1 = diff[:,:,:,1]
    #diff_2 = diff[:,:,:,2]
    #diff_3 = diff[:,:,:,3]
    #sum_min_dist_0 = torch.sum(torch.min(torch.norm(diff_0, dim=-1,p=2), dim = -1).values)
    #sum_min_dist_1 = torch.sum(torch.min(torch.norm(diff_1, dim=-1,p=2), dim = -1).values)
    #sum_min_dist_2 = torch.sum(torch.min(torch.norm(diff_2, dim=-1,p=2), dim = -1).values)
    #sum_min_dist_3 = torch.sum(torch.min(torch.norm(diff_3, dim=-1,p=2), dim = -1).values)
    #print(sum_min_dist_0,sum_min_dist_1,sum_min_dist_2,sum_min_dist_3)

    dist = torch.norm(diff, dim=-1,p=2) #x1 - y1 + eps

    # For every output value, find its closest input value; for every input value, find its closest output value.
    min_dist_xy = torch.min(dist, dim = -1)  # Get min distance per row - Find the closest input to the output
    min_dist_yx = torch.min(dist, dim = -2)  # Get min distance per column - Find the closest output to the input
    eucl =  1./2*torch.sum(min_dist_xy.values + min_dist_yx.values)
    batch_size = x.shape[0]
    num_particles = x.shape[1]
    eucl =  eucl/(batch_size*num_particles)

    xy_idx = min_dist_xy.indices.clone()
    yx_idx = min_dist_yx.indices.clone() 

    aux_idx = num_particles * torch.arange(batch_size).to(device) #We create auxiliary indices to separate per batch of particles
    aux_idx = aux_idx.view(batch_size, 1)
    aux_idx = torch.repeat_interleave(aux_idx, num_particles, axis=-1)
    xy_idx = xy_idx + aux_idx
    yx_idx = yx_idx + aux_idx

    xy_idx = xy_idx.reshape((batch_size*num_particles))
    yx_idx = yx_idx.reshape((batch_size*num_particles))
    return  eucl, xy_idx, yx_idx

    
def categorical_loss_cosine(target, reco,xy_idx, yx_idx,loss_fnc):
    get_x = target[xy_idx]
    get_y = reco[yx_idx]

    #reco : get_x #reco - the closest input  to the output 
    #target : get_y # :target - the closest output to the input
    #first argument NN output, second argument is true labels
    mask = Variable(torch.ones(get_y.shape[0]), requires_grad=False).to(device)
    loss = 1./2.*(loss_fnc(reco,get_x,mask) + loss_fnc(get_y,target,mask))

    return loss

def categorical_loss(target, reco,xy_idx, yx_idx,loss_fnc):
    get_x = target[xy_idx].long()
    get_y = reco[yx_idx]

    #reco : get_x #reco - the closest input  to the output 
    #target : get_y # :target - the closest output to the input
    #first argument NN output, second argument is true labels

    loss = loss_fnc(reco,get_x) + loss_fnc(get_y,target.long()) #output loss per graph node
    return loss


def huber_loss(target, reco,xy_idx, yx_idx):
    get_x = target[xy_idx]
    get_y = reco[yx_idx]

    #loss_fnc = nn.MSELoss()
    loss_fnc = torch.nn.HuberLoss(delta=10.0)
    loss = 2*(loss_fnc(reco,get_x) + loss_fnc(get_y,target)) #2* because in Huber loss there is a factor 1/2
    return loss


class CustomHuberLoss:
    def __init__(self, delta = 1.0, reduction ='mean' ):
        self.delta = delta
        self.reduction = reduction

    def __call__(self, difference):
        errors = torch.abs(difference)

        mask = errors < self.delta
        res = (0.5 * mask * (torch.pow(errors,2))) + ~mask *(self.delta*(errors-0.5*self.delta))
        if self.reduction=='mean':
            return torch.mean(res)
        else:
            return torch.sum(res)

def global_met_loss(target, reco):
    #this custom class is not needed anymore, we can just use directly Huber. it was needed for applying cyclic operation on delta phi
    diff = target-reco

    loss_fnc = CustomHuberLoss(delta=10.0)
    loss = 2*(loss_fnc(diff)) #2* because in Huber loss there is a factor 1/2
    return loss


def pairwise_distance(x, y):
    if (x.shape[0] != y.shape[0]):
        raise ValueError("The batch size of x and y are not equal! x.shape[0] is {}, whereas y.shape[0] is {}!".fromat(x.shape[0],y.shape[0]))
    if (x.shape[-1] != y.shape[-1]):
        raise ValueError("Feature dimension of x and y are not equal! x.shape[-1] is {}, whereas y.shape[-1] is {}!".format(x.shape[-1],y.shape[-1]))


    batch_size = x.shape[0]
    num_row = x.shape[1]
    num_col = y.shape[1]
    vec_dim = x.shape[-1]

    x1 = x.repeat(1, 1, num_col).view(batch_size, -1, num_col, vec_dim).to(device)
    y1 = y.repeat(1, num_row, 1).view(batch_size, num_row, -1, vec_dim).to(device)

    diff = x1 - y1
    return diff    


class EdgeConvLayer(nn.Module):   
    def __init__(self, in_dim, out_dim, dropout=0.0, batch_norm=True, act=nn.LeakyReLU(negative_slope=0.3), aggr='mean'):
        super().__init__()
        self.activation = act
        self.batch_norm = batch_norm
            
        if self.batch_norm:
            self.edgeconv = nn.Sequential(nn.Linear(2*(in_dim), out_dim),
                                   nn.BatchNorm1d(out_dim),
                                   self.activation,
                                   nn.Dropout(p=dropout)) 
        else :
            self.edgeconv = nn.Sequential(nn.Linear(2*(in_dim), out_dim),
                                   self.activation,
                                   nn.Dropout(p=dropout))             

        ###dropout in AE as a regularization 
        self.edgeconv = EdgeConv(nn=self.edgeconv,aggr=aggr)

    def forward(self, feature, edge_index):
        h = self.edgeconv(feature, edge_index)
    
        return h

class GraphAE(torch.nn.Module):
  def __init__(self, input_shape, hidden_channels,latent_dim,activation=nn.LeakyReLU(negative_slope=0.1),input_shape_global = 2):
    super(GraphAE, self).__init__()

    #Which main block to use for the architecture
    layer = EdgeConvLayer # EdgeConvLayer #GCNConv #GCN
    layer_kwargs = {'act':activation}
    if layer == GCN :
        layer_kwargs['num_layers'] = 1

    self.activation = activation 
    #self.batchnorm = nn.BatchNorm1d(input_shape[-1])
    self.num_fixed_nodes = input_shape[0]
    self.input_shape_global = input_shape_global
    self.hidden_global = 4*input_shape_global
    self.num_feats = input_shape[1]
    self.num_pid_classes = 4
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
    #x_phi = cycle_by_2pi(x_phi)
    #x_phi = PI*torch.tanh(x_phi)
    #x_eta = 5.*torch.tanh(x_eta) # have a symmetric activation function that eventually dies out. 

#########
    x_phi = 7.*torch.tanh(x_phi)
    x_eta = 6.*torch.tanh(x_eta)
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

    def forward(self, data):
        x_bar, x_met_bar, z = self.ae(data)
        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x_bar, x_met_bar, q, z

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
        pred_labels = np.array([self.forward(d.to(device))[2].data.cpu().numpy().argmax(1) for i,d in enumerate(test_loader)]) #argmax(1) #index (cluster nubmber) of the cluster with the highest probability q.
        latent_pred = np.array([self.forward(d.to(device))[3].data.cpu().numpy() for i,d in enumerate(test_loader)])
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
        total_loss, total_reco_loss, total_pid_loss, total_energy_loss, total_met_loss = 0.,0.,0.,0.,0.
        for i, data in enumerate(train_loader):
            data = data.to(device)
            x = data.x.to(device)
            x_met = data.x_met.reshape((-1,model.input_shape_global)).to(device)
            batch_index = data.batch.to(device)

            optimizer.zero_grad()
            x_bar,x_met_bar, z = model(data)
            #x_embedded = model.embedded_input
            #loss = F.mse_loss(x_bar, x) 
            #loss, xy_idx, yx_idx = chamfer_loss(x_embedded,x_bar,batch_index)
            #reco_loss, xy_idx, yx_idx  = chamfer_loss(x[:,1:],x_bar[:,model.ae.num_pid_classes:],batch_index)
            #reco_loss, xy_idx, yx_idx  = chamfer_loss(x[:,[model.eta_idx,model.phi_idx]],x_bar[:,[model.num_pid_classes-1 + model.eta_idx, model.num_pid_classes-1 + model.phi_idx]],batch_index)

            reco_loss, xy_idx, yx_idx  = chamfer_loss(x[:,1:],x_bar[:,model.num_pid_classes:],batch_index)

            met_loss = global_met_loss(x_met, x_met_bar)

            #energy_loss  = huber_loss(x[:,[model.energy_idx,model.pt_idx]],x_bar[:,[model.num_pid_classes-1 + model.energy_idx, model.num_pid_classes-1 + model.pt_idx]],xy_idx, yx_idx)

            nll_loss = nn.NLLLoss(reduction='mean',weight=pid_weight)
            pid_loss = categorical_loss(x[:,0],x_bar[:,0:model.num_pid_classes],xy_idx, yx_idx,nll_loss) #target, reco

            loss = reco_loss  + pid_loss_weight*pid_loss + met_loss_weight*met_loss #+ energy_loss_weight*energy_loss
            total_loss += loss.item()
            total_reco_loss += reco_loss.item()
            total_pid_loss += pid_loss.item()
            total_met_loss += met_loss.item()
            #total_energy_loss += energy_loss.item()
            total_energy_loss += 0. 

            loss.backward()
            optimizer.step()

        print("epoch {} : total loss={:.4f}, reco loss={:.4f}, pid loss={:.4f}, energy loss={:.4f}, met loss={:.4f}  ".format(epoch, total_loss / (i + 1) , total_reco_loss / (i + 1), total_pid_loss/(i+1), total_energy_loss/(i+1),total_met_loss/(i+1)))

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
            data = data.to(device)
            _,_, hidden = model.ae(data)
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
            data = data.to(device)
            model.clustering(mbk,data)

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
        total_loss,total_kl_loss,total_reco_loss,total_pid_loss,total_energy_loss,total_met_loss  = 0.,0.,0.,0.,0.,0.
        for i, data in enumerate(train_loader):
            data = data.to(device)
            x = data.x.to(device)
            x_met = data.x_met.reshape((-1,model.ae.input_shape_global)).to(device)
            batch_index = data.batch.to(device)

            x_bar, x_met_bar, q, _ = model(data)
            #x_embedded = model.ae.embedded_input

            #reconstr_loss = F.mse_loss(x_bar, x)
            #reconstr_loss, xy_idx, yx_idx  = chamfer_loss(x[:,1:],x_bar[:,model.ae.num_pid_classes:],batch_index)
            #cos_emb_loss = torch.nn.CosineEmbeddingLoss(reduction='mean')
            #pid_loss = categorical_loss(x[:,0:1],x_bar[:,0:1],xy_idx, yx_idx,cos_emb_loss)
            #nll_loss = nn.NLLLoss(reduction='mean',weight=pid_weight)
            #pid_loss = categorical_loss(x[:,0],x_bar[:,0:model.ae.num_pid_classes],xy_idx, yx_idx,nll_loss) #target, reco


            #reco_loss, xy_idx, yx_idx  = chamfer_loss(x[:,[model.ae.eta_idx,model.ae.phi_idx]],x_bar[:,[model.ae.num_pid_classes-1 + model.ae.eta_idx, model.ae.num_pid_classes-1 + model.ae.phi_idx]],batch_index)
            
            reco_loss, xy_idx, yx_idx  = chamfer_loss(x[:,1:],x_bar[:,model.ae.num_pid_classes:],batch_index)

            met_loss = global_met_loss(x_met, x_met_bar)

            #energy_loss  = huber_loss(x[:,[model.ae.energy_idx,model.ae.pt_idx]],x_bar[:,[model.ae.num_pid_classes-1 + model.ae.energy_idx, model.ae.num_pid_classes-1 + model.ae.pt_idx]],xy_idx, yx_idx)

            nll_loss = nn.NLLLoss(reduction='mean',weight=pid_weight)
            pid_loss = categorical_loss(x[:,0],x_bar[:,0:model.ae.num_pid_classes],xy_idx, yx_idx,nll_loss) #target, reco

            kl_loss = F.kl_div(q.log(), p_all[i],reduction='batchmean')
            loss = args.gamma * kl_loss + reco_loss + pid_loss_weight*pid_loss + met_loss_weight*met_loss #+ energy_loss_weight*energy_loss 

            optimizer.zero_grad()
            total_loss += loss.item()
            total_kl_loss += kl_loss.item()
            total_reco_loss += reco_loss.item()
            total_pid_loss += pid_loss.item()
            #total_energy_loss += energy_loss.item()
            total_energy_loss += 0.
            total_met_loss = met_loss.item()

            loss.backward()
            optimizer.step()
        print("epoch {} : total loss={:.4f}, kl loss={:.4f}, reco loss={:.4f}, pid loss={:.4f}, energy loss={:.4f}, met loss={:.4f} ".format(epoch, total_loss / (i + 1), total_kl_loss / (i + 1) , total_reco_loss / (i + 1), total_pid_loss/(i+1), total_energy_loss/(i+1), total_met_loss/(i+1)))
        if epoch==args.n_epochs-1 :
            torch.save(model.state_dict(), args.pretrain_path.replace('.pkl','_fullmodel_num_clust_{}.pkl'.format(args.n_clusters)))
        torch.save(model.state_dict(), args.pretrain_path.replace('.pkl','_fullmodel_num_clust_{}.pkl'.format(args.n_clusters)))




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
    parser.add_argument('--input_shape', default=[16,5], type=int)
    parser.add_argument('--hidden_channels', default=[8, 12, 16, 20, 25, 30, 40, 60,50,40,30 ], type=int)   ## [8, 12, 16, 20, 25, 30 ]
    parser.add_argument('--pretrain_path', type=str, default='data_graph/graph_ae_pretrain.pkl') 
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
    TRAIN_NAME = 'background_chan3_passed_ae_l1.h5'
    filename_bg = DATA_PATH + TRAIN_NAME 
    in_file = h5py.File(filename_bg, 'r') 
    file_dataset = np.array(in_file['dataset'])
    file_dataset[:,:,2] = file_dataset[:,:,2]/1e5 #E
    file_dataset[:,:,3] = file_dataset[:,:,3]/1e5 #pT
    #log of energy and pt as preprocessing
    file_dataset[:,:,2] = np.log(file_dataset[:,:,2]+1)
    file_dataset[:,:,3] = np.log(file_dataset[:,:,3]+1) 
    file_dataset[:,1:,4] = np.where(file_dataset[:,1:,1]==0.,0.,file_dataset[:,1:,4]+3.0)
    file_dataset[:,1:,5] = np.where(file_dataset[:,1:,1]==0.,0.,file_dataset[:,1:,5]+3.4)

    #Select top N processes only :
    n_proc = 3
    (unique, counts) = np.unique(file_dataset[:,:,0], return_counts=True)
    procs_sorted, counts_sorted = zip(*sorted(zip(unique, counts), key=lambda x: x[1],reverse=True))
    top_proc_mask = np.isin(file_dataset[:,0,0], procs_sorted[:n_proc]) #choose top 3
    file_dataset = file_dataset[top_proc_mask][:,:args.input_shape[0]+1,:]


    datas = []
    tot_evt = file_dataset.shape[0]# int(1e4)
    print('Preparing the dataset of {} events'.format(tot_evt))
    n_objs = args.input_shape[0]
    #connecting all particles
    #adj = [csr_matrix(np.ones((n_objs,n_objs)) - np.eye(n_objs))]*tot_evt
    #edge_index = [from_scipy_sparse_matrix(a)[0] for a in adj]   

    #conencting only real particles
    adj_non_con = make_adjacencies(file_dataset[:,1:,])
    adj_non_connected = [csr_matrix(adj_non_con[i]) for i in range(tot_evt)]
    edge_index = [from_scipy_sparse_matrix(a)[0] for a in adj_non_connected]      

    x = [torch.tensor(file_dataset[i_evt,1:,1:], dtype=torch.float) for i_evt in range(tot_evt)]
    y = [torch.tensor(int(file_dataset[i_evt,0,0]), dtype=torch.int) for i_evt in range(tot_evt)]
    x_met = [torch.tensor(file_dataset[i_evt,0,[2,5]], dtype=torch.float) for i_evt in range(tot_evt)]
    datas = [Data(x=x_event, edge_index=edge_index_event,y=torch.unsqueeze(y_event, 0),x_met=x_met_event) 
    for x_event,edge_index_event,y_event,x_met_event in zip(x,edge_index,y,x_met)]
    print('Dataset of {} events prepared'.format(tot_evt))
    dataset  = GraphDataset(datas)
    #pid_weight = torch.tensor([1./0.52,1./0.37,1./0.025,1./0.016,1./0.06]).to(device)
    #pid_weight = torch.tensor([1.,1.4,20.,30.,9.]).to(device)
    #pid_weight = torch.tensor([1.,1.4,10.,10.,10.]).to(device)
    pid_weight = torch.tensor([1.,1.4,5.,5.]).to(device)
    #pid_weight = torch.tensor([1.3,1.,12.,19.]).to(device) #4 clasess without met , 11 particles
    #pid_weight = torch.tensor([5,1.,12.,19.]).to(device) #4 clasess without met, 7 particles
    #pid_weight = torch.tensor([1.3,1.,7.,5.]).to(device) #4 clasess without met , 11 particles

    energy_loss_weight = 1.0
    pid_loss_weight = 0.1
    met_loss_weight = 10.0

    print(args)
    train_idec()
