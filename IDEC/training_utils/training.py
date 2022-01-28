# Code developed by © chernyavskaya 
# Starting point for iDEC from © dawnranger
#
import numpy as np

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader as DataLoaderTorch

from torch_geometric.data import Data, Batch, DataLoader

from training_utils.metrics import cluster_acc
from training_utils.losses import chamfer_loss,huber_mask,categorical_loss,huber_loss,global_met_loss


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_test_ae_graph(model,loader,optimizer,device,pid_weight,pid_loss_weight,met_loss_weight,energy_loss_weight,mode='test'):
    if mode=='test':
        model.eval()
    else:
        model.train()

    total_loss, total_reco_loss, total_pid_loss, total_energy_loss, total_met_loss = 0.,0.,0.,0.,0.
    for i, data in enumerate(loader):
        data = data.to(device)
        x = data.x.to(device)
        x_met = data.x_met.reshape((-1,model.input_shape_global)).to(device)
        batch_index = data.batch.to(device)

        if mode=='train':
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

        nll_loss = torch.nn.NLLLoss(reduction='mean',weight=pid_weight)
        pid_loss = categorical_loss(x[:,0],x_bar[:,0:model.num_pid_classes],xy_idx, yx_idx,nll_loss) #target, reco

        loss = reco_loss  + pid_loss_weight*pid_loss + met_loss_weight*met_loss #+ energy_loss_weight*energy_loss
        total_loss += loss.item()
        total_reco_loss += reco_loss.item()
        total_pid_loss += pid_loss.item()
        total_met_loss += met_loss.item()
        #total_energy_loss += energy_loss.item()
        total_energy_loss += 0. 

        if mode=='train':
            loss.backward()
            optimizer.step()

    return total_loss / (i + 1) , total_reco_loss / (i + 1), total_pid_loss/(i+1), total_energy_loss/(i+1),total_met_loss/(i+1)


def train_test_idec_graph(model,loader,p_all,optimizer,device,gamma,pid_weight,pid_loss_weight,met_loss_weight,energy_loss_weight,mode='test'):
    if mode=='test':
        model.eval()
    else:
        model.train()

    total_loss, total_kl_loss,total_reco_loss, total_pid_loss, total_energy_loss, total_met_loss = 0.,0., 0.,0.,0.,0.
    for i, data in enumerate(loader):
        data = data.to(device)
        x = data.x.to(device)
        x_met = data.x_met.reshape((-1,model.ae.input_shape_global)).to(device)
        batch_index = data.batch.to(device)

        if mode=='train':
            optimizer.zero_grad()
        x_bar,x_met_bar, q, _ = model(data)
        #x_embedded = model.embedded_input
        #loss = F.mse_loss(x_bar, x) 
        #loss, xy_idx, yx_idx = chamfer_loss(x_embedded,x_bar,batch_index)
        #reco_loss, xy_idx, yx_idx  = chamfer_loss(x[:,1:],x_bar[:,model.ae.num_pid_classes:],batch_index)
        #reco_loss, xy_idx, yx_idx  = chamfer_loss(x[:,[model.ae.eta_idx,model.ae.phi_idx]],x_bar[:,[model.ae.num_pid_classes-1 + model.ae.eta_idx, model.ae.num_pid_classes-1 + model.ae.phi_idx]],batch_index)

        reco_loss, xy_idx, yx_idx  = chamfer_loss(x[:,1:],x_bar[:,model.ae.num_pid_classes:],batch_index)

        met_loss = global_met_loss(x_met, x_met_bar)

        #energy_loss  = huber_loss(x[:,[model.ae.energy_idx,model.ae.pt_idx]],x_bar[:,[model.ae.num_pid_classes-1 + model.ae.energy_idx, model.ae.num_pid_classes-1 + model.ae.pt_idx]],xy_idx, yx_idx)

        nll_loss = torch.nn.NLLLoss(reduction='mean',weight=pid_weight)
        pid_loss = categorical_loss(x[:,0],x_bar[:,0:model.ae.num_pid_classes],xy_idx, yx_idx,nll_loss) #target, reco

        kl_loss = F.kl_div(q.log(), p_all[i],reduction='batchmean')
        loss = gamma * kl_loss + reco_loss + pid_loss_weight*pid_loss + met_loss_weight*met_loss #+ energy_loss_weight*energy_loss 

        loss = reco_loss  + pid_loss_weight*pid_loss + met_loss_weight*met_loss #+ energy_loss_weight*energy_loss
        total_loss += loss.item()
        total_kl_loss += kl_loss.item()
        total_reco_loss += reco_loss.item()
        total_pid_loss += pid_loss.item()
        total_met_loss += met_loss.item()
        #total_energy_loss += energy_loss.item()
        total_energy_loss += 0. 

        if mode=='train':
            loss.backward()
            optimizer.step()

    return total_loss / (i + 1) , total_kl_loss / (i+1), total_reco_loss / (i + 1), total_pid_loss/(i+1), total_energy_loss/(i+1),total_met_loss/(i+1)



def pretrain_ae_graph(model,train_loader,test_loader,optimizer,n_epochs,pretrain_path,device,pid_weight,pid_loss_weight,met_loss_weight,energy_loss_weight):
    '''
    pretrain autoencoder for graph
    '''
    for epoch in range(n_epochs):
        train_loss , train_reco_loss, train_pid_loss, train_energy_loss, train_met_loss = train_test_ae_graph(model,train_loader,optimizer,device,pid_weight,pid_loss_weight,met_loss_weight,energy_loss_weight,mode='train')
        test_loss , test_reco_loss, test_pid_loss, test_energy_loss, test_met_loss = train_test_ae_graph(model,test_loader,optimizer,device,pid_weight,pid_loss_weight,met_loss_weight,energy_loss_weight,mode='test')

        print("epoch {} : TRAIN : total loss={:.4f}, reco loss={:.4f}, pid loss={:.4f}, energy loss={:.4f}, met loss={:.4f}  ".format(epoch, train_loss , train_reco_loss, train_pid_loss, train_energy_loss, train_met_loss))
        print("epoch {} : TEST : total loss={:.4f}, reco loss={:.4f}, pid loss={:.4f}, energy loss={:.4f}, met loss={:.4f}  ".format(epoch,test_loss , test_reco_loss, test_pid_loss, test_energy_loss, test_met_loss ))

    torch.save(model.state_dict(), pretrain_path)
    print("model saved to {}.".format(pretrain_path))



def train_test_ae_dense(model,loader,optimizer,device,mode='test'):
    if mode=='test':
        model.eval()
    else:
        model.train()

    total_loss = 0.
    for i, (x,_) in enumerate(loader):
        x = x.to(device)

        if mode=='train':
            optimizer.zero_grad()
        x_bar,_, z = model(x)

        loss = huber_mask(x,x_bar)

        total_loss += loss.item()
        #print(total_loss)

        if mode=='train':
            loss.backward()
            optimizer.step()
    return total_loss / (i + 1)


def train_test_idec_dense(model,loader,p_all,optimizer,device,gamma,mode='test'):
    if mode=='test':
        model.eval()
    else:
        model.train()

    total_loss,total_reco_loss,total_kl_loss  = 0.,0.,0.
    for i, (x,_) in enumerate(loader):
        x = x.to(device)

        if mode=='train':
            optimizer.zero_grad()
        x_bar,_, q ,_ = model(x)

        loss = huber_mask(x,x_bar)

        reco_loss = huber_mask(x,x_bar)
        kl_loss = F.kl_div(q.log(), p_all[i],reduction='batchmean')
        loss = gamma * kl_loss + reco_loss 

        total_loss += loss.item()
        total_reco_loss += reco_loss.item()
        total_kl_loss += kl_loss.item()

        if mode=='train':
            loss.backward()
            optimizer.step()
    return total_loss / (i + 1),  total_kl_loss / (i + 1), total_reco_loss / (i + 1)




def pretrain_ae_dense(model,train_loader,test_loader,optimizer,n_epochs,pretrain_path,device):
    '''
    pretrain autoencoder for dense
    '''
    for epoch in range(n_epochs):

        train_loss = train_test_ae_dense(model,train_loader,optimizer,device,mode='train')
        test_loss = train_test_ae_dense(model,test_loader,optimizer,device,mode='test')

        print("epoch {} : TRAIN : total loss={:.4f}".format(epoch, train_loss ))
        print("epoch {} : TEST : total loss={:.4f}".format(epoch, test_loss ))

    torch.save(model.state_dict(),pretrain_path)
    print("model saved to {}.".format(pretrain_path))

