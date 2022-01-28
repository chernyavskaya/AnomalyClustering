# Developed by © chernyavskaya 
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
from torch.optim import Adam

from torch_geometric.data import Data, Batch, DataLoader
from torch.utils.data import DataLoader as DataLoaderTorch

from data_utils.data_processing import GraphDataset, DenseEventDataset
from training_utils.metrics import cluster_acc
from models.models import DenseAE, IDEC 
from training_utils.training import pretrain_ae_dense,train_test_ae_dense,train_test_idec_dense

import os.path as osp
sys.path.append(os.path.abspath(os.path.join('../../')))
sys.path.append(os.path.abspath(os.path.join('../')))
import ADgvae.utils_torch.model_summary as summary

#torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_idec():

    model_AE = 
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

    if args.retrain_ae :
        model.pretrain()
    else :
        model.pretrain(args.pretrain_path)

    train_loader = DataLoaderTorch(
        dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    optimizer = Adam(model.parameters(), lr=args.lr)

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
    for i, (x,_) in enumerate(kmeans_loader):
        x = x.to(device)
        model.clustering(mbk,x,args.full_kmeans,kmeans_initialized)

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

            pred_labels = np.array([self.forward(x.to(device))[2].data.cpu().numpy().argmax(1) for i,(x,_) in enumerate(train_loader)]) #argmax(1) ##index (cluster nubmber) of the cluster with the highest probability q.
            latent_pred = np.array([self.forward(x.to(device))[3].data.cpu().numpy() for i,(x,_) in enumerate(train_loader)])
            true_labels = np.array([y.cpu().numpy() for i,(_,y) in enumerate(train_loader)])

            #The evaluation of accuracy is done for information only, we cannot use the true labels for the training 
            acc, nmi, ari, _,  = model.validateOnCompleteTestData(true_labels,pred_labels,latent_pred)
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
        train_loss,train_reco_loss,train_kl_loss = train_test_idec_dense(model,train_loader,p_all,optimizer,device,gamma,mode='train')
        test_loss,test_reco_loss,test_kl_loss = train_test_idec_dense(model,test_loader,p_all,optimizer,device,gamma,mode='test')

        print("epoch {} : TRAIN : total loss={:.4f}, reco loss={:.4f}, kl loss={:.4f} ".format(epoch, train_loss, train_reco_loss, train_kl_loss ))
        print("epoch {} : TEST : total loss={:.4f}, reco loss={:.4f}, kl loss={:.4f} ".format(epoch, test_loss, test_reco_loss, test_kl_loss ))
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
    parser.add_argument('--input_shape', default=[124], type=int)
    parser.add_argument('--hidden_channels', default=[50,30,10], type=int)   
    parser.add_argument('--pretrain_path', type=str, default='data_dense/dense_ae_pretrain.pkl') 
    parser.add_argument('--gamma',default=100.,type=float,help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--n_epochs',default=100, type=int)
    args = parser.parse_args()

    DATA_PATH = '/eos/user/n/nchernya/MLHEP/AnomalyDetection/AnomalyClustering/inputs/'
    TRAIN_NAME = 'bkg_sig_0.0156_l1_filtered_padded.h5'
    filename_bg = DATA_PATH + TRAIN_NAME 
    in_file = h5py.File(filename_bg, 'r') 
    file_dataset = np.array(in_file['dataset'])
    file_dataset_1d,file_dataset_proc_truth = prepare_1d_datasets(file_dataset)

    dataset = DenseEventDataset(file_dataset_1d,file_dataset_proc_truth)

    print(args)
    train_idec()
