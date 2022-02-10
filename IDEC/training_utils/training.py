# Code developed by © chernyavskaya 
# Starting point for iDEC from © dawnranger
#
import numpy as np
import tqdm
import shutil
import os, json

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader as DataLoaderTorch

from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.utils import to_dense_batch

from training_utils.metrics import cluster_acc
from training_utils.losses import chamfer_loss,huber_mask,categorical_loss,huber_loss,global_met_loss,chamfer_loss_split, chamfer_loss_split_parallel
from training_utils.plot_losses import loss_curves
from tensorboard.backend.event_processing import event_accumulator
import multiprocessing as mp


def merge_loss_dicts(dict_list):
    merged_dict = {}
    if len(dict_list)==0:
        return {}
    for key in dict_list[0]:
        merged_dict[key] = dict_list[0][key]
        for i_dict in range(1,len(dict_list)):
            merged_dict[key] = list(merged_dict[key])
            if not (set(dict_list[i_dict][key][0]) <= set(merged_dict[key][0])):
                merged_dict[key][0] += dict_list[i_dict][key][0]
                merged_dict[key][1] += dict_list[i_dict][key][1]
    return merged_dict


def export_jsondump(writer):

    assert isinstance(writer, torch.utils.tensorboard.SummaryWriter)

    data_dict_list=[]
    tf_files = [] # -> list of paths from writer.log_dir to all files in that directory
    for root, dirs, files in os.walk(writer.log_dir):
        for file in files:
            if 'events.out.tfevents' in file:
                tf_files.append(os.path.join(root,file)) # go over every file recursively in the directory

    for file_id, file in enumerate(tf_files):

        path = os.path.join('/'.join(file.split('/')[:-1])) # determine path to folder in which file lies
        name = 'data_'+str(file_id)
        #os.path.join(file.split('/')[-2]) if file_id > 0 else os.path.join('data') # seperate file created by add_scalar from add_scalars

        # print(file, '->', path, '|', name)

        event_acc = event_accumulator.EventAccumulator(file)
        event_acc.Reload()
        data = {}

        hparam_file = False # I save hparam files as 'hparam/xyz_metric'
        for tag in sorted(event_acc.Tags()["scalars"]):
            if tag.split('/')[0] == 'hparam': hparam_file=True # check if its a hparam file
            step, value = [], []

            for scalar_event in event_acc.Scalars(tag):
                step.append(scalar_event.step)
                value.append(scalar_event.value)

            data[tag] = (step, value)

        if data!={} : 
            data_dict_list.append(data)
        if not hparam_file and bool(data): # if its not a hparam file and there is something in the data -> dump it
            with open(path+f'/{name}.json', "w") as f:
                json.dump(data, f)
    merged_data = merge_loss_dicts(data_dict_list)
    with open(path+f'/data_merged.json', "w") as f:
        json.dump(merged_data, f)
    return merged_data


def save_ckp(state,checkpoint_path):
    """
    state: checkpoint to save
    checkpoint_path: path to save checkpoint
    """
    f_path = checkpoint_path
    torch.save(state, f_path)

def load_ckp(checkpoint_fpath, model, optimizer=None, scheduler=None):
    """
    checkpoint_path: path from where to load checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer to load
    scheduler: scheduler to load
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['model'])
    # initialize optimizer and scheduler from checkpoint 
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    # initialize test_loss and and train_loss from checkpoint 
    train_loss = checkpoint['train_loss']
    test_loss = checkpoint['test_loss']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, scheduler, checkpoint['epoch'], train_loss, test_loss


def create_ckp(epoch, train_loss,test_loss,model_state_dict,optimizer_state_dict, scheduler_state_dict):
    # create checkpoint variable and add important data
    checkpoint = {
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'test_loss': test_loss,
        'model': model_state_dict,
        'optimizer': optimizer_state_dict,
        'scheduler': scheduler_state_dict,
        }
    return checkpoint


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_test_ae_graph(model,loader,optimizer,device,pid_weight,pid_loss_weight,met_loss_weight,energy_loss_weight,mode='test'):
    if mode=='test':
        model.eval()
    else:
        model.train()

    total_loss, total_reco_loss, total_pid_loss, total_energy_loss, total_met_loss, total_reco_zero_loss = 0.,0.,0.,0.,0.,0.
    t = tqdm.tqdm(enumerate(loader),total=len(loader))
    pool = mp.Pool(mp.cpu_count())
    for i, data in t:
        data = data.to(device)
        x = data.x.to(device)
        x_met = data.x_met.reshape((-1,model.input_shape_global)).to(device)
        batch_index = data.batch.to(device)

        if mode=='train':
            optimizer.zero_grad()
            x_bar,x_met_bar, z = model(data)
        else:
            with torch.no_grad():
                x_bar,x_met_bar, z = model(data)

        #x_embedded = model.embedded_input
        #loss = F.mse_loss(x_bar, x) 
        #loss, xy_idx, yx_idx = chamfer_loss(x_embedded,x_bar,batch_index)
        #reco_loss, xy_idx, yx_idx  = chamfer_loss(x[:,1:],x_bar[:,model.ae.num_pid_classes:],batch_index)
        #reco_loss, xy_idx, yx_idx  = chamfer_loss(x[:,[model.eta_idx,model.phi_idx]],x_bar[:,[model.num_pid_classes-1 + model.eta_idx, model.num_pid_classes-1 + model.phi_idx]],batch_index)

        #reco_loss, xy_idx, yx_idx  = chamfer_loss(x[:,1:],x_bar[:,model.num_pid_classes:],batch_index)

        #use chamefer only to get indecies first 
        _, xy_idx, yx_idx  = chamfer_loss(x[:,1:],x_bar[:,model.num_pid_classes:],batch_index)

        met_loss = global_met_loss(x_met, x_met_bar)

        #energy_loss  = huber_loss(x[:,[model.energy_idx,model.pt_idx]],x_bar[:,[model.num_pid_classes-1 + model.energy_idx, model.num_pid_classes-1 + model.pt_idx]],xy_idx, yx_idx)

        nll_loss = torch.nn.NLLLoss(reduction='mean',weight=pid_weight)
        pid_loss = categorical_loss(x[:,0],x_bar[:,0:model.num_pid_classes],xy_idx, yx_idx,nll_loss) #target, reco


        target_dense = to_dense_batch(x[:,1:], batch_index)[0].detach().cpu().numpy()
        reco_dense = to_dense_batch(x_bar[:,model.num_pid_classes:], batch_index)[0].detach().cpu().numpy()
        in_pid_dense = to_dense_batch(x[:,0], batch_index)[0].detach().cpu().numpy()
        out_pid_dense = to_dense_batch(x_bar[:,0:model.num_pid_classes].argmax(1), batch_index)[0].detach().cpu().numpy()
        reco_loss, reco_zero_loss =  chamfer_loss_split_parallel(pool,target_dense, reco_dense, in_pid_dense, out_pid_dense)
        #reco_loss, reco_zero_loss  = torch.tensor(reco_loss).to(device), torch.tensor(reco_zero_loss).to(device) 
        #reco_loss, reco_zero_loss =  chamfer_loss_split(x[:,1:],x_bar[:,model.num_pid_classes:],x[:,0],x_bar[:,0:model.num_pid_classes],batch_index)



        loss = reco_loss +reco_zero_loss + pid_loss_weight*pid_loss + met_loss_weight*met_loss #+ energy_loss_weight*energy_loss
        total_loss += loss.item()
        total_reco_loss += reco_loss.item()
        total_reco_zero_loss  += reco_zero_loss.item()
        total_pid_loss += pid_loss.item()
        total_met_loss += met_loss.item()
        #total_energy_loss += energy_loss.item()
        total_energy_loss += 0. 


        if mode=='train':
            loss.backward()
            optimizer.step()

    pool.close()
    i = len(loader)
    return total_loss / (i + 1) , total_reco_loss / (i + 1), total_pid_loss/(i+1), total_energy_loss/(i+1),total_met_loss/(i+1), total_reco_zero_loss / (i + 1)


def train_test_idec_graph(model,loader,p_all,optimizer,device,gamma,pid_weight,pid_loss_weight,met_loss_weight,energy_loss_weight,mode='test'):
    if mode=='test':
        model.eval()
    else:
        model.train()

    total_loss, total_kl_loss,total_reco_loss, total_pid_loss, total_energy_loss, total_met_loss,total_reco_loss = 0.,0., 0.,0.,0.,0.,0.
    t = tqdm.tqdm(enumerate(loader),total=len(loader))
    for i, data in t:
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

        #reco_loss, xy_idx, yx_idx  = chamfer_loss(x[:,1:],x_bar[:,model.ae.num_pid_classes:],batch_index)
        _, xy_idx, yx_idx  = chamfer_loss(x[:,1:],x_bar[:,model.ae.num_pid_classes:],batch_index)

        met_loss = global_met_loss(x_met, x_met_bar)

        #energy_loss  = huber_loss(x[:,[model.ae.energy_idx,model.ae.pt_idx]],x_bar[:,[model.ae.num_pid_classes-1 + model.ae.energy_idx, model.ae.num_pid_classes-1 + model.ae.pt_idx]],xy_idx, yx_idx)

        nll_loss = torch.nn.NLLLoss(reduction='mean',weight=pid_weight)
        pid_loss = categorical_loss(x[:,0],x_bar[:,0:model.ae.num_pid_classes],xy_idx, yx_idx,nll_loss) #target, reco


        #target_dense = to_dense_batch(x[:,1:], batch_index)[0].detach().cpu().numpy()
        #reco_dense = to_dense_batch(x_bar[:,model.ae.num_pid_classes:], batch_index)[0].detach().cpu().numpy()
        #in_pid_dense = to_dense_batch(x[:,0], batch_index)[0].detach().cpu().numpy()
        #out_pid_dense = to_dense_batch(x_bar[:,0:model.ae.num_pid_classes].argmax(1), batch_index)[0].detach().cpu().numpy()
        #reco_loss, reco_zero_loss =  chamfer_loss_split_parallel(target_dense, reco_dense, in_pid_dense, out_pid_dense)
        #reco_loss, reco_zero_loss  = torch.tensor(reco_loss).to(device), torch.tensor(reco_zero_loss).to(device) 
        reco_loss, reco_zero_loss =  chamfer_loss_split(x[:,1:],x_bar[:,model.ae.num_pid_classes:],x[:,0],x_bar[:,0:model.ae.num_pid_classes],batch_index)

        kl_loss = F.kl_div(q.log(), p_all[i],reduction='batchmean')
        loss = gamma * kl_loss + (1.-gamma)*(reco_loss + reco_zero_loss+ pid_loss_weight*pid_loss + met_loss_weight*met_loss) #+ energy_loss_weight*energy_loss 

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
    i = len(loader)
    return total_loss / (i + 1) , total_kl_loss / (i+1), total_reco_loss / (i + 1), total_pid_loss/(i+1), total_energy_loss/(i+1),total_met_loss/(i+1)



def pretrain_ae_graph(model,train_loader,test_loader,optimizer,start_epoch,n_epochs,pretrain_path,device,scheduler,summary_writer,pid_weight,pid_loss_weight,met_loss_weight,energy_loss_weight):
    '''
    pretrain autoencoder for graph
    '''
    best_test_loss=10000.
    for epoch in range(start_epoch,n_epochs):
        train_loss , train_reco_loss, train_pid_loss, train_energy_loss, train_met_loss,train_reco_zero_loss = train_test_ae_graph(model,train_loader,optimizer,device,pid_weight,pid_loss_weight,met_loss_weight,energy_loss_weight,mode='train')
        test_loss , test_reco_loss, test_pid_loss, test_energy_loss, test_met_loss,test_reco_zero_loss = train_test_ae_graph(model,test_loader,optimizer,device,pid_weight,pid_loss_weight,met_loss_weight,energy_loss_weight,mode='test')

        print("epoch {} : TRAIN : total loss={:.4f}, reco loss={:.4f}, pid loss={:.4f}, energy loss={:.4f}, met loss={:.4f}, reco zero loss={:.4f}  ".format(epoch, train_loss , train_reco_loss, train_pid_loss, train_energy_loss, train_met_loss,train_reco_zero_loss))
        print("epoch {} : TEST : total loss={:.4f}, reco loss={:.4f}, pid loss={:.4f}, energy loss={:.4f}, met loss={:.4f}, reco zero loss={:.4f}  ".format(epoch,test_loss , test_reco_loss, test_pid_loss, test_energy_loss, test_met_loss,test_reco_zero_loss ))
        scheduler.step(test_loss)

        loss_names = ["Loss Tot","Loss Reco","Loss Pid","Loss Energy","Loss Met","Loss Reco Zero"]
        for name, loss in zip(loss_names,[train_loss , train_reco_loss, train_pid_loss, train_energy_loss, train_met_loss,train_reco_zero_loss]):
            summary_writer.add_scalar("Training "+ name, loss, epoch)
        for name, loss in zip(loss_names,[test_loss , test_reco_loss, test_pid_loss, test_energy_loss, test_met_loss,test_reco_zero_loss]):
            summary_writer.add_scalar("Validation "+ name, loss, epoch)   
        for layer_name, weight in model.named_parameters():
            summary_writer.add_histogram(layer_name,weight, epoch)
            summary_writer.add_histogram(f'{layer_name}.grad',weight.grad, epoch)

        if epoch>10 and test_loss < best_test_loss*1.01: #allow variation within 1%
            best_test_loss = test_loss
            best_fpath = pretrain_path.replace(pretrain_path.rsplit('/', 1)[-1],'')+'best_model_AE.pkl'
            checkpoint = create_ckp(epoch, train_loss,test_loss,model.state_dict(),optimizer.state_dict(), scheduler.state_dict())
            save_ckp(checkpoint, best_fpath)
            print('New best model saved')
        if epoch>10 and epoch%10==0:
            checkpoint = create_ckp(epoch, train_loss,test_loss,model.state_dict(),optimizer.state_dict(), scheduler.state_dict())
            save_ckp(checkpoint, pretrain_path.replace('.pkl','_epoch_{}.pkl'.format(epoch+1)))

    checkpoint = create_ckp(epoch, train_loss,test_loss,model.state_dict(),optimizer.state_dict(), scheduler.state_dict())
    save_ckp(checkpoint, pretrain_path.replace('.pkl','_epoch_{}.pkl'.format(epoch+1)))
    print("model saved to {}.".format(pretrain_path))
    merged_data = export_jsondump(summary_writer)
    loss_curves(merged_data, pretrain_path.replace(pretrain_path.rsplit('/', 1)[-1],'')+'/fig_dir/ae/')
    summary_writer.close()



def train_test_ae_dense(model,loader,optimizer,device,mode='test'):
    if mode=='test':
        model.eval()
    else:
        model.train()

    total_loss = 0.
    t = tqdm.tqdm(enumerate(loader),total=len(loader))
    for i, (x,_) in t:

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
    i = len(loader)
    return total_loss / (i + 1)


def train_test_idec_dense(model,loader,p_all,optimizer,device,gamma,mode='test'):
    if mode=='test':
        model.eval()
    else:
        model.train()

    total_loss,total_reco_loss,total_kl_loss  = 0.,0.,0.
    t = tqdm.tqdm(enumerate(loader),total=len(loader))
    for i, (x,_) in t:

        x = x.to(device)

        if mode=='train':
            optimizer.zero_grad()
        x_bar,_, q ,_ = model(x)

        reco_loss = huber_mask(x,x_bar)
        kl_loss = F.kl_div(q.log(), p_all[i],reduction='batchmean')
        loss = gamma * kl_loss + (1.-gamma)*reco_loss 

        total_loss += loss.item()
        total_reco_loss += reco_loss.item()
        total_kl_loss += kl_loss.item()

        if mode=='train':
            loss.backward()
            optimizer.step()
    i = len(loader)
    return total_loss / (i + 1),  total_kl_loss / (i + 1), total_reco_loss / (i + 1)




def pretrain_ae_dense(model,train_loader,test_loader,optimizer,start_epoch,n_epochs,pretrain_path,device,scheduler, summary_writer):
    '''
    pretrain autoencoder for dense
    '''
    best_test_loss=10000.
    for epoch in range(start_epoch,n_epochs):

        train_loss = train_test_ae_dense(model,train_loader,optimizer,device,mode='train')
        test_loss = train_test_ae_dense(model,test_loader,optimizer,device,mode='test')

        print("epoch {} : TRAIN : total loss={:.4f}".format(epoch, train_loss ))
        print("epoch {} : TEST : total loss={:.4f}".format(epoch, test_loss ))

        scheduler.step(test_loss)

        summary_writer.add_scalar("Training Loss Tot", train_loss, epoch)
        summary_writer.add_scalar("Validation Loss Tot", test_loss, epoch)
        for layer_name, weight in model.named_parameters():
            summary_writer.add_histogram(layer_name,weight, epoch)
            summary_writer.add_histogram(f'{layer_name}.grad',weight.grad, epoch)

        if epoch>10 and test_loss < best_test_loss*1.01: #allow variation within 1%
            best_test_loss = test_loss
            best_fpath = pretrain_path.replace(pretrain_path.rsplit('/', 1)[-1],'')+'best_model_AE.pkl'
            checkpoint = create_ckp(epoch, train_loss,test_loss,model.state_dict(),optimizer.state_dict(), scheduler.state_dict())
            save_ckp(checkpoint, best_fpath)
            print('New best model saved')
        if epoch>10 and epoch%10==0:
            checkpoint = create_ckp(epoch, train_loss,test_loss,model.state_dict(),optimizer.state_dict(), scheduler.state_dict())
            save_ckp(checkpoint, pretrain_path.replace('.pkl','_epoch_{}.pkl'.format(epoch+1)))

    checkpoint = create_ckp(epoch, train_loss,test_loss,model.state_dict(),optimizer.state_dict(), scheduler.state_dict())
    save_ckp(checkpoint, pretrain_path.replace('.pkl','_epoch_{}.pkl'.format(epoch+1)))
    print("model saved to {}.".format(pretrain_path))
    merged_data = export_jsondump(summary_writer)
    loss_curves(merged_data, pretrain_path.replace(pretrain_path.rsplit('/', 1)[-1],'')+'/fig_dir/ae/')
    summary_writer.close()
