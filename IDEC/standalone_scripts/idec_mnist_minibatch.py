# -*- coding: utf-8 -*-
#
# Copyright © dawnranger.
#
# 2018-05-08 10:15 <dawnranger123@gmail.com>
#
# Distributed under terms of the MIT license.
from __future__ import print_function, division
import setGPU
import os
import argparse
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear

from utils import MnistDataset, cluster_acc

class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()

        # encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)

        self.z_layer = Linear(n_enc_3, n_z)

        # decoder
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)

        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):

        # encoder
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))

        z = self.z_layer(enc_h3)

        # decoder
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z


class IDEC(nn.Module):

    def __init__(self,
                 n_enc_1,
                 n_enc_2,
                 n_enc_3,
                 n_dec_1,
                 n_dec_2,
                 n_dec_3,
                 n_input,
                 n_z,
                 n_clusters,
                 alpha=1,
                 pretrain_path='data/ae_mnist.pkl'):
        super(IDEC, self).__init__()
        self.alpha = 1.0
        self.pretrain_path = pretrain_path

        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def pretrain(self, path=''):
        if path == '':
            pretrain_ae(self.ae)
        # load pretrain weights
        self.ae.load_state_dict(torch.load(self.pretrain_path))
        self.ae.to(device)
        print('load pretrained ae from', path)

    def forward(self, x):

        x_bar, z = self.ae(x)
        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x_bar, q, z

    def clustering(self,mbk,x):
        self.eval()
        _,pred_labels_ae = self.ae(x)
        pred_labels_ae = pred_labels_ae.data.cpu().numpy()
        pred_labels = mbk.partial_fit(pred_labels_ae) #seems we can only get a centre from batch
        self.cluster_centers = mbk.cluster_centers_ #keep the cluster centers
        self.updateClusterCenter(self.cluster_centers)

    def updateClusterCenter(self, cc):
        self.cluster_layer.data = torch.from_numpy(cc).to(device)

    def validateOnCompleteTestData(self,test_loader):
        self.eval()
        pred_labels = np.array([self.forward(d[0].to(device))[1].data.cpu().numpy().argmax(1) for i,d in enumerate(test_loader)]) #argmax(1) #index (cluster nubmber) of the cluster with the highest probability q.
        true_labels = np.array([d[1].cpu().numpy() for i,d in enumerate(test_loader)])
        #reshape
        pred_labels = np.reshape(pred_labels,pred_labels.shape[0]*pred_labels.shape[1])
        true_labels = np.reshape(true_labels,true_labels.shape[0]*true_labels.shape[1])

        acc,_ = cluster_acc(true_labels, pred_labels)
        nmi = nmi_score(true_labels, pred_labels)
        ari = ari_score(true_labels, pred_labels)
        return acc, nmi, ari, pred_labels

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def pretrain_ae(model):
    '''
    pretrain autoencoder
    '''
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,drop_last=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(200):
        total_loss = 0.
        for batch_idx, (x, _, _) in enumerate(train_loader):
            x = x.to(device)

            optimizer.zero_grad()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("epoch {} loss={:.4f}".format(epoch,
                                            total_loss / (batch_idx + 1)))
        torch.save(model.state_dict(), args.pretrain_path)
    print("model saved to {}.".format(args.pretrain_path))


def train_idec():

    model = IDEC(
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=1000,
        n_dec_1=1000,
        n_dec_2=500,
        n_dec_3=500,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        alpha=1.0,
        pretrain_path=args.pretrain_path)
    model.to(device)


    model.pretrain('data/ae_mnist.pkl')
    #model.pretrain()

    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    optimizer = Adam(model.parameters(), lr=args.lr)

    print('Initializing cluster center with pre-trained weights')
##############################
    #Mini batch k-means if k-means on full dataset cannot fit in memory, fast but slower convergence of the model as cluster initialization is not as good
    mbk = MiniBatchKMeans(n_clusters=args.n_clusters, n_init=20, batch_size=args.batch_size)
    #step 1 - get cluster center from batch
    #here we are using minibatch kmeans to be able to cope with larger dataset.
    for batch_idx, (x, _, _) in enumerate(train_loader):
        x = x.to(device)
        model.clustering(mbk,x)

###############################
#    #Full k-means if dataset can fit in memeory (better cluster initialization and faster convergence of the model)
#    data = dataset.x
#    y = dataset.y
#    data = torch.Tensor(data).to(device)
#    _, hidden = model.ae(data)
#    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
#    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
#    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
##############################

    pred_labels_last = 0
    delta_label = 1e4
    model.train()
    for epoch in range(100):

        #evaluation part
        if epoch % args.update_interval == 0:

            p_all = []
            for batch_idx, (x, _, idx) in enumerate(train_loader):
                x = x.to(device)

                _, tmp_q_i, _ = model(x)
                tmp_q_i = tmp_q_i.data
                p = target_distribution(tmp_q_i)
                p_all.append(p)

            acc, nmi, ari, pred_labels = model.validateOnCompleteTestData(train_loader)
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
        for batch_idx, (x, _, _) in enumerate(train_loader):

            x = x.to(device)
            idx = idx.to(device)

            x_bar, q, _ = model(x)

            reconstr_loss = F.mse_loss(x_bar, x)
            kl_loss = F.kl_div(q.log(), p_all[batch_idx])
            loss = args.gamma * kl_loss + reconstr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch==99 :
            torch.save(model.state_dict(), args.pretrain_path.replace('.pkl','_fullmodel_num_clust_{}.pkl'.format(args.n_clusters)))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--pretrain_path', type=str, default='data/ae_mnist')
    parser.add_argument(
        '--gamma',
        default=0.1,
        type=float,
        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = "cpu"

    if args.dataset == 'mnist':
        args.pretrain_path = 'data/ae_mnist.pkl'
        #args.n_clusters = 10 #try different
        args.n_input = 784
        dataset = MnistDataset()
    print(args)
    train_idec()
