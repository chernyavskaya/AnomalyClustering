import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.autograd import Variable

from torch_geometric.utils import from_scipy_sparse_matrix, to_dense_batch

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

