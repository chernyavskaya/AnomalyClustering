import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.autograd import Variable

from torch_geometric.utils import from_scipy_sparse_matrix, to_dense_batch
import multiprocessing as mp



eps = 1e-12
PI = math.pi
TWOPI = 2*math.pi


def cycle_by_2pi(in_tensor):
    in_tensor  = torch.where(in_tensor >= PI, in_tensor-TWOPI, in_tensor)
    in_tensor  = torch.where(in_tensor < -PI, in_tensor+TWOPI, in_tensor)
    return in_tensor


def huber_mask(inputs, outputs):
    #the masking is already applied at the level of output nodes
    #input_zeros_mask = torch.ne(inputs,0).float().to(inputs.device)
    #outputs= input_zeros_mask * outputs
    # we might want to introduce different weighting to the parts of the loss with  jets/muons/...
    loss_fnc = torch.nn.HuberLoss(delta=10.0)
    loss = loss_fnc(inputs,outputs)
    return loss

def chamfer_loss_split_parallel(pool,target, reco, in_pid, out_pid):
    n_batches = target.shape[0]

    eucl_list = np.array([pool.apply(chamfer_loss_split_per_batch_element, args=(target[i], reco[i], in_pid[i], out_pid[i])) for i in range(0,n_batches)])
    #pool.close()  
    reduce_sum = eucl_list.sum(axis=0)
    return reduce_sum[0]/n_batches,reduce_sum[1]/n_batches


def chamfer_loss_split_pid_parallel(pool,target, reco, in_pid, out_pid,pids):
    n_batches = target.shape[0]

    eucl_list = np.array([pool.apply(chamfer_loss_split_pid_per_batch_element, args=(target[i], reco[i], in_pid[i], out_pid[i],pids)) for i in range(0,n_batches)])
    #pool.close()  
    reduce_sum = eucl_list.sum(axis=0)/n_batches
    #return total loss for non 0 particles (summed over all pid), and 0.
    return np.sum(reduce_sum[1:]),reduce_sum[0]


def chamfer_loss_split_per_batch_element(x_ib, y_ib, in_pid_dense, out_pid_dense):
    eucl_non_zero = 0.
    eucl_zero = 0.

    #construct masks here based on pid 
    input_non_zeros_mask = np.not_equal(in_pid_dense,0)
    output_non_zeros_mask = np.not_equal(out_pid_dense,0)
    x_non_zero = x_ib[input_non_zeros_mask]
    y_non_zero = y_ib[output_non_zeros_mask]
    n_in_part = max(1,x_non_zero.shape[0]) # to avoid dividing by 0
    n_out_part = max(1,y_non_zero.shape[0]) 
    if len(y_non_zero)==0 :
        eucl_non_zero+=np.sum(np.sqrt(np.sum(x_non_zero**2,axis=-1)))/n_in_part
    elif len(x_non_zero)==0 :
        eucl_non_zero+=np.sum(np.sqrt(np.sum(y_non_zero**2,axis=-1)))/n_out_part
    else:
        n_in_part = x_non_zero.shape[0]
        n_out_part = y_non_zero.shape[0]
        diff_non_zero = np.expand_dims(x_non_zero,1) - np.expand_dims(y_non_zero,0)
        dist_non_zero =np.sqrt(np.sum(diff_non_zero**2,axis=-1))
        #eucl for all particles that are not 0 , normal chamfer 
        min_dist_xy = np.min(dist_non_zero, axis=-1)
        min_dist_yx = np.min(dist_non_zero, axis=-2)
        eucl_non_zero +=  1./2*(np.sum(min_dist_xy)/n_out_part + np.sum(min_dist_yx)/n_in_part)

    y_zero = y_ib[~output_non_zeros_mask]
    n_out_part = max(1,y_zero.shape[0])
    eucl_zero += np.sum(np.sqrt(np.sum(y_zero**2,axis=-1)))/n_out_part
    return [eucl_non_zero, eucl_zero]



def chamfer_loss_split_pid_per_batch_element(x_ib, y_ib, in_pid_dense, out_pid_dense,pids):
    eucl_non_zero = 0.
    eucl_zero = 0.
    #pids 
    eucl_losses = np.zeros(len(pids))
    #loop with real particles (pid != 0)
    for pid in pids[1:]:
        #construct masks here based on pid 
        input_pid_mask = np.equal(in_pid_dense,pid)
        output_pid_mask = np.equal(out_pid_dense,pid)
        x_masked = x_ib[input_pid_mask]
        y_masked = y_ib[output_pid_mask]
        n_in_part = max(1,x_masked.shape[0]) # to avoid dividing by 0
        n_out_part = max(1,y_masked.shape[0]) 
        #Devision per particle is needed because we have different occurence and multiplicity for different pid
        if len(y_masked)==0 :
            eucl_losses[pid]+=np.sum(np.sqrt(np.sum(x_masked**2,axis=-1)))/n_in_part
        elif len(x_masked)==0 :
            eucl_losses[pid]+=np.sum(np.sqrt(np.sum(y_masked**2,axis=-1)))/n_out_part
        else:
            diff = np.expand_dims(x_masked,1) - np.expand_dims(x_masked,0)
            dist =np.sqrt(np.sum(diff**2,axis=-1))
            #eucl for all particles that are not 0 , normal chamfer 
            min_dist_xy = np.min(dist, axis=-1)
            min_dist_yx = np.min(dist, axis=-2)
            eucl_losses[pid] +=  1./2*(np.sum(min_dist_xy)/n_out_part + np.sum(min_dist_yx)/n_in_part)

    #handle zeros :
    output_zeros_mask = np.equal(out_pid_dense,0)
    y_zero = y_ib[output_zeros_mask]
    #n_out_part = max(1,y_zero.shape[0])
    #Zeros :  do we want to divide here by n_out_part. Maybe not. 
    eucl_losses[0] += 10*np.sum(np.sqrt(np.sum(y_zero**2,axis=-1))) #/n_out_part
    return eucl_losses


def chamfer_loss_split(target, reco, in_pid, out_pid, batch):
    #In principle we can even do prediction per PID the same way.. but we can only do this if we manage to speed up the implementation
    #non zero particles : 
    n_batches = 0
    eucl_non_zero = 0.
    eucl_zero = 0.
    #256 x 17 x 4 -> numpy , paralilize it 
    for ib in torch.unique(batch):
        x_ib = target[batch==ib].to(target.device)
        y_ib = reco[batch==ib].to(target.device)
        #construct masks here based on pid 
        input_non_zeros_mask = torch.ne(in_pid[batch==ib],0).to(target.device)
        output_non_zeros_mask = torch.ne(out_pid[batch==ib].argmax(1),0).to(reco.device)
        x_non_zero = x_ib[input_non_zeros_mask].to(target.device)
        y_non_zero = y_ib[output_non_zeros_mask].to(target.device)
        #add a check that checks if y is not empty. if y is empty then loss should be sum of x_non_zero : torch.sum(torch.norm(x_non_zero,dim=-1,p=2))
        if len(y_non_zero)==0 and len(x_non_zero)!=0:
            eucl_non_zero+=torch.sum(torch.norm(x_non_zero,dim=-1,p=2))
        else:
            n_in_part = x_non_zero.shape[0]
            n_out_part = y_non_zero.shape[0]
            diff_non_zero = pairwise_distance_per_item(x_non_zero,y_non_zero).to(target.device)
            dist_non_zero = torch.norm(diff_non_zero, dim=-1,p=2).to(target.device)
            #eucl for all particles that are not 0 , normal chamfer 
            min_dist_xy = torch.min(dist_non_zero, dim = -1)
            min_dist_yx = torch.min(dist_non_zero, dim = -2)
            eucl_non_zero +=  1./2*(torch.sum(min_dist_xy.values)/n_out_part + torch.sum(min_dist_yx.values)/n_in_part)

        y_zero = y_ib[~output_non_zeros_mask].to(target.device)
        eucl_zero += torch.sum(torch.norm(y_zero,dim=-1,p=2))

        n_batches+=1
    eucl_non_zero /= n_batches  
    eucl_zero /= n_batches

    return  eucl_non_zero,eucl_zero

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

    aux_idx = num_particles * torch.arange(batch_size).to(target.device) #We create auxiliary indices to separate per batch of particles
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
    mask = Variable(torch.ones(get_y.shape[0]), requires_grad=False).to(target.device)
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

    x1 = x.repeat(1, 1, num_col).view(batch_size, -1, num_col, vec_dim).to(x.device)
    y1 = y.repeat(1, num_row, 1).view(batch_size, num_row, -1, vec_dim).to(x.device)

    diff = x1 - y1
    return diff  

def pairwise_distance_per_item(x, y):
    #implementation of pairwise distance for each item in a batch. Needed when each batch has different length of x and y, e.g. after masking

    num_row = x.shape[0]
    num_col = y.shape[0]
    vec_dim = x.shape[-1]

    x1 = x.repeat(1, 1, num_col).view(-1, num_col, vec_dim).to(x.device)
    y1 = y.repeat(1, num_row, 1).view(num_row, -1, vec_dim).to(x.device)

    diff = x1 - y1
    return diff   

def pairwise_distance_per_item_numpy(x, y):
    #implementation of pairwise distance for each item in a batch. Needed when each batch has different length of x and y, e.g. after masking
    distances = np.subtract(x[:,np.newaxis,:],y[np.newaxis,:,:])
    return distances


