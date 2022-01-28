import numpy as np
import torch
from torch.utils.data import Dataset
from  torch_geometric.data import Dataset as PyGDataset

#def load_mnist(path='./IDEC_pytorch/data/mnist.npz'):
def load_mnist(path='./data/mnist.npz'):

    f = np.load(path)

    x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f[
        'x_test'], f['y_test']
    f.close()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test)).astype(np.int32)
    x = x.reshape((x.shape[0], -1)).astype(np.float32)
    x = np.divide(x, 255.)
    print('MNIST samples', x.shape)
    return x, y



class DenseEventDataset(Dataset):

    def __init__(self,dataset_1d,dataset_truth):
        super().__init__()
        self.dataset_1d, self.dataset_truth = dataset_1d,dataset_truth

    def __len__(self):
        return self.dataset_1d.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.dataset_1d[idx])).float(), torch.from_numpy(
            np.array(self.dataset_truth[idx])).float()


class MnistDataset(Dataset):

    def __init__(self):
        self.x, self.y = load_mnist()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))


class GraphDataset(PyGDataset):
    def __init__(self, datas, transform=None, pre_transform=None):
        super().__init__(transform=None, pre_transform=None)

        self.datas=datas 

    def len(self):
        return len(self.datas)

    def get_data(self):
        self.x = torch.from_numpy(np.stack([self.datas[i].x for i in range(len(self.datas))],axis=0)) 
        self.y = torch.from_numpy(np.stack([self.datas[i].y for i in range(len(self.datas))],axis=0))
        self.y = torch.squeeze(self.y,dim=-1)
        self.edge_index = torch.from_numpy(np.stack([self.datas[i].edge_index for i in range(len(self.datas))],axis=0))
        return self.x, self.y, self.edge_index

    def get(self, idx):
        return self.datas[idx]




def get_relative_weights(data,mode='max'):
    unique, counts = np.unique(data, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    if mode=='max':
        weight = np.max(frequencies[:,-1])/frequencies[:,-1]
    elif mode=='sum':
        weight = frequencies[:,-1]/np.sum(frequencies[:,-1])
    else :
        print('Mode {} is unknown. The available options are : max and sum'.format(mode))
    return weight

def make_adjacencies(particles):
    real_p_mask = particles[:,:,1] > 0. # construct mask for real particles
    adjacencies = (real_p_mask[:,:,np.newaxis] * real_p_mask[:,np.newaxis,:]).astype('float32')
    return adjacencies

def proprocess_e_pt(file_dataset, idx=[2,3], scale=1.e5,log=True):
    file_dataset[:,:,idx] = file_dataset[:,:,idx]/scale 
    #log of energy and pt as preprocessing
    if log==True:
        file_dataset[:,:,idx] = np.log(file_dataset[:,:,idx]+1)
    return file_dataset

def select_top_n_procs(file_dataset,n_proc):
    #Select top N processes only :
    (unique, counts) = np.unique(file_dataset[:,:,0], return_counts=True)
    procs_sorted, counts_sorted = zip(*sorted(zip(unique, counts), key=lambda x: x[1],reverse=True))
    top_proc_mask = np.isin(file_dataset[:,0,0], procs_sorted[:n_proc]) #choose top 3
    return top_proc_mask

def prepare_1d_datasets(file_dataset,n_top_proc = -1):
    print('Preparing dataset, check that the feature indexing corresponds to your dataset!')

    if n_top_proc > 0:
        top_proc_mask =  select_top_n_procs(file_dataset,n_top_proc)
        file_dataset = file_dataset[top_proc_mask]

    proc_idx = 0
    dkl_idx = 1
    charge_idx = 2
    features_idx = [3,4,5,6]

    dataset_proc_truth = file_dataset[:,0,[proc_idx]]
    dataset_2d = file_dataset[:,:,features_idx]
    #scale and take log of energy and pt as preprocessing
    dataset_2d = proprocess_e_pt(dataset_2d,idx=[0,1],scale=1e5,log=True)
    #set met pt and eta as 0
    dataset_2d[:,0,1] = 0. #met pt is 0
    dataset_2d[:,0,2] = 0. #met eta is 0

    #dataset_met = dataset_2d[:,0,[0,-1]] #met is first , and only has E and phi
    #dataset_particles = dataset_2d[:,1:,:] #other particles follow , and have all features 
    #dataset_particles = dataset_particles.reshape((dataset_particles.shape[0],dataset_particles.shape[1]*dataset_particles.shape[2]))
    #dataset_1d = np.hstack([dataset_met,dataset_particles])
    
    dataset_1d = dataset_2d.reshape((dataset_2d.shape[0],dataset_2d.shape[1]*dataset_2d.shape[2]))
    dense_dataset = DenseEventDataset(dataset_1d,dataset_proc_truth)
    return dataset_1d,dataset_proc_truth,dense_dataset


def prepare_graph_datas(file_dataset,n_particles,n_top_proc = -1,connect_only_real=True):
    print('Preparing dataset, check that the feature indexing corresponds to your dataset!')
    datas = []
    file_dataset = proprocess_e_pt(file_dataset,idx=[2,3],scale=1e5,log=True)

    if n_top_proc > 0:
        top_proc_mask =  select_top_n_procs(file_dataset,n_top_proc)
        file_dataset = file_dataset[top_proc_mask]
    file_dataset = file_dataset[:,:n_particles+1,:]

    tot_evt = file_dataset.shape[0]
    print('Preparing the dataset of {} events'.format(tot_evt))
    n_objs = n_particles
    if not connect_only_real:
        #connecting all particles
        adj = [csr_matrix(np.ones((n_objs,n_objs)) - np.eye(n_objs))]*tot_evt
        edge_index = [from_scipy_sparse_matrix(a)[0] for a in adj]   
    else:
        #connecting only real particles
        adj_non_con = make_adjacencies(file_dataset[:,1:,])
        adj_non_connected = [csr_matrix(adj_non_con[i]) for i in range(tot_evt)]
        edge_index = [from_scipy_sparse_matrix(a)[0] for a in adj_non_connected]      

    x = [torch.tensor(file_dataset[i_evt,1:,1:], dtype=torch.float) for i_evt in range(tot_evt)]
    y = [torch.tensor(int(file_dataset[i_evt,0,0]), dtype=torch.int) for i_evt in range(tot_evt)]
    x_met = [torch.tensor(file_dataset[i_evt,0,[2,5]], dtype=torch.float) for i_evt in range(tot_evt)]
    datas = [Data(x=x_event, edge_index=edge_index_event,y=torch.unsqueeze(y_event, 0),x_met=x_met_event) 
    for x_event,edge_index_event,y_event,x_met_event in zip(x,edge_index,y,x_met)]
    print('Dataset of {} events prepared'.format(tot_evt))
    graph_dataset  = GraphDataset(datas)
    return datas,graph_dataset





