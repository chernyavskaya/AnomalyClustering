import os,sys
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Dataset as PyGDataset
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_dense_batch
from scipy.sparse import csr_matrix
import random
import h5py

def load_mnist(path='../data/mnist.npz'):

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


class GraphDatasetOnline(PyGDataset):
    def __init__(self,root,input_files,datasetname,truth_datasetname,n_events=-1,data_chunk_size=2e4,input_shape=[18,5],connect_only_real=True, shuffle=False,transform=None, pre_transform=None):
        super().__init__(transform=None, pre_transform=None)
        """
        Initialize parameters of the graph dataset
        Args:
            root (str) : path for input files
            input_files (list) : input h5 files
            datasetname (str) : dataset name inside h5 file
            truth_datasetname (str) : dataset name for the truth (process id) inside h5 file
            n_events (int): how many events to process (-1=all in a file)
            data_chunk_size (int) : chunk of teh dataset to load in memmory
            input_shape (list) : input shape of the particles graph
            connect_only_real (bool) : connect all particles or only existing
            shuffle (bool): shuffle or not 
        """
        self.strides = [0]
        self.len_in_files = []
        self.input_files = input_files
        self.datasetname = datasetname
        self.truth_datasetname = truth_datasetname
        self.shuffle = shuffle 
        self.n_events = int(n_events)
        self.connect_only_real = connect_only_real
        self.input_shape = input_shape
        super(GraphDatasetOnline, self).__init__(root, transform, pre_transform)
        self.calculate_offsets()
        self.data_chunk_size=int(data_chunk_size)
        self.current_file_idx=0
        self.current_in_file = h5py.File(self.processed_file_names[self.current_file_idx],'r')#,driver='core',backing_store=False)
        self.current_chunk_idx = 0
        self.current_pytorch_datas = self.in_memory_data(shuffle=self.shuffle)


    @property 
    def processed_dir(self) -> str: #keep it as is in pytorch 
        return self.root


    @property
    def processed_file_names(self):
        proc_list = [self.root + '/' + f for f in self.input_files] 
        return proc_list


    def len(self):
        if self.n_events <= self.strides[-1]:
            return self.n_events
        else:
            return self.strides[-1]

    def calculate_offsets(self):
        for path in self.processed_paths:
            with h5py.File(path, 'r') as f:
                self.strides.append(f[self.datasetname].shape[0])
        self.len_in_files = self.strides[1:]
        self.strides = np.cumsum(self.strides)
        if self.n_events==-1:
            self.n_events = self.strides[-1]


    def get_data_from_file(self):
        n_start = self.current_chunk_idx*self.data_chunk_size
        n_end = n_start+self.data_chunk_size 
        if n_end > self.len_in_files[self.current_file_idx]:
            n_end = self.len_in_files[self.current_file_idx]
        return self.get_data_from_file_basic(n_start=n_start,n_end=n_end,infile=self.current_in_file)


    def get_data_from_file_basic(self,n_start=0,n_end=0,infile=None):
        if infile==None:
            print('No file is passed for processing, exiting')
            exit()

        file_dataset = prepare_ad_event_based_dataset(np.array(infile[self.datasetname])[n_start:n_end,:,:],truth_dataset=np.array(infile[self.truth_datasetname])[n_start:n_end])
        return file_dataset


    def in_memory_data(self,shuffle=False,file_dataset=None,return_prepared_array=False): 
        if file_dataset==None :
            file_dataset = self.get_data_from_file() 
        prepared_dataset,datas =  prepare_graph_datas(file_dataset,self.input_shape[0],connect_only_real=self.connect_only_real)
        if shuffle:
            random.Random(0).shuffle(datas)
            random.Random(0).shuffle(prepared_dataset)
        if return_prepared_array :
            return prepared_dataset,datas
        else :
            return datas


    def get(self, idx):
        """ Used by PyTorch DataSet class """    
        file_idx = np.searchsorted(self.strides, idx,side='right') - 1
        idx_in_file = idx - self.strides[max(0, file_idx)] 
        if file_idx >= self.strides.size:
            raise Exception(f'{idx} is beyond the end of the event list {self.strides[-1]}')
        if self.current_file_idx != file_idx:
            self.current_file_idx = file_idx
            self.current_in_file.close()
            self.current_in_file = h5py.File(self.processed_paths[self.current_file_idx],'r')#,driver='core',backing_store=False)
            self.current_chunk_idx = 0 #reset current chunk index
            self.current_pytorch_datas = self.in_memory_data(shuffle=self.shuffle)
        if (idx_in_file >= (self.current_chunk_idx+1)*self.data_chunk_size): 
            self.current_chunk_idx+=1
            self.current_pytorch_datas = self.in_memory_data(shuffle=self.shuffle)
        idx_in_chunk =  idx_in_file % len(self.current_pytorch_datas)
        element_to_yield = self.current_pytorch_datas[idx_in_chunk]
        #Finally reset everything at the end of an epoch
        if idx==self.len()-1:
            if self.current_file_idx!=0:
                self.current_file_idx=0
                self.current_in_file.close()
                self.current_in_file = h5py.File(self.processed_paths[self.current_file_idx],'r')#,driver='core',backing_store=False)
            self.current_chunk_idx = 0
            self.current_pytorch_datas =self.in_memory_data(shuffle=self.shuffle)
        return element_to_yield
        



def get_relative_weights(data,mode='max'):
    unique, counts = np.unique(data, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    if mode=='max':
        weight = np.max(frequencies[:,-1])/frequencies[:,-1]
    elif mode=='sum':
        weight = frequencies[:,-1]/np.sum(frequencies[:,-1])
    else :
        print('Mode {} is unknown. The available options are : max and sum'.format(mode))
    return weight.astype(float)

def make_adjacencies(particles):
    real_p_mask = particles[:,:,1] > 0. # construct mask for real particles
    adjacencies = (real_p_mask[:,:,np.newaxis] * real_p_mask[:,np.newaxis,:]).astype('float32')
    return adjacencies

def proprocess_e_pt(file_dataset, idx=[2,3], scale=1.e5,shift=0.,log=True):
    if len(file_dataset.shape)==3:
        file_dataset[:,:,idx] = file_dataset[:,:,idx]/scale 
        #log of energy and pt as preprocessing, but create a continious distribution 
        if log==True:
            #file_dataset[:,:,idx] = np.log(file_dataset[:,:,idx]+1)
            file_dataset[:,:,idx] = np.where(file_dataset[:,:,idx]>0,\
                                            np.log(file_dataset[:,:,idx]+1e-6),\
                                            shift)
            file_dataset[:,:,idx] -=shift
    elif len(file_dataset.shape)==2:
        file_dataset[:,idx] = file_dataset[:,idx]/scale 
        #log of energy and pt as preprocessing
        if log==True:
            #file_dataset[:,idx] = np.log(file_dataset[:,idx]+1)
            file_dataset[:,idx] = np.where(file_dataset[:,idx]>0,\
                                            np.log(file_dataset[:,idx]+1e-6),\
                                            shift) #np.log(np.min(file_dataset[:,idx][file_dataset[:,idx]>0]))*0.999
            file_dataset[:,idx] -=shift

    return file_dataset


def select_top_n_procs(file_dataset_proc,n_top_proc):
    #Select top N processes only :
    (unique, counts) = np.unique(file_dataset_proc, return_counts=True)
    procs_sorted, counts_sorted = zip(*sorted(zip(unique, counts), key=lambda x: x[1],reverse=True))
    top_proc_mask = np.isin(file_dataset_proc, procs_sorted[:n_top_proc])
    return top_proc_mask

def prepare_1d_reduced_datasets(file_dataset,n_top_proc = -1):
    print('Preparing dataset, check that the feature indexing corresponds to your dataset!')

    if n_top_proc > 0:
        top_proc_mask =  select_top_n_procs(file_dataset[:,0],n_top_proc)
        file_dataset = file_dataset[top_proc_mask]

    proc_idx = 0

    dataset_proc_truth = file_dataset[:,[proc_idx]]
    dataset_particles = file_dataset[:,1:]
    dataset_particles = proprocess_e_pt(dataset_particles,idx=[0,2,6],scale=1e5,log=True) #[MET, N(e/ɣ/μ), max pT(e/ɣ/μ), η(e/ɣ/μ), ɸ(e/ɣ/μ), N(jets), max pT(jets), η(jets), ɸ(jets) ]]
    dataset_met = np.zeros((dataset_particles.shape[0],4))
    dataset_met[:,1] = np.copy(dataset_particles[:,0])
    dataset_particles = dataset_particles[:,1:]
    dataset_1d = np.hstack([dataset_met,dataset_particles])    
    dense_dataset = DenseEventDataset(dataset_1d,dataset_proc_truth)
    return dataset_1d,dataset_proc_truth,dense_dataset


def prepare_1d_datasets(file_dataset,n_top_proc = -1):
    print('Preparing dataset, check that the feature indexing corresponds to your dataset!')

    if n_top_proc > 0:
        top_proc_mask =  select_top_n_procs(file_dataset[:,0,0],n_top_proc)
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
    #print('Preparing dataset, check that the feature indexing corresponds to your dataset!')
    datas = []
    #file_dataset = proprocess_e_pt(file_dataset,idx=[2,3],scale=1e5,log=True) #idx=[2,3]
    file_dataset = proprocess_e_pt(file_dataset,idx=[2],scale=1.,shift=np.log(4.)*0.999,log=True) #idx=[1] 

    if n_top_proc > 0:
        top_proc_mask =  select_top_n_procs(file_dataset[:,0,0],n_top_proc)
        file_dataset = file_dataset[top_proc_mask]
    file_dataset = file_dataset[:,:n_particles+1,:]

    tot_evt = file_dataset.shape[0]
    #print('Preparing the dataset of {} events'.format(tot_evt))
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
    #x_met = [torch.tensor(file_dataset[i_evt,0,[2,5]], dtype=torch.float) for i_evt in range(tot_evt)]
    x_met = [torch.tensor(file_dataset[i_evt,0,[2,4,5]], dtype=torch.float) for i_evt in range(tot_evt)]
    datas = [Data(x=x_event, edge_index=edge_index_event,y=torch.unsqueeze(y_event, 0),x_met=x_met_event) 
    for x_event,edge_index_event,y_event,x_met_event in zip(x,edge_index,y,x_met)]
    #print('Dataset of {} events prepared'.format(tot_evt))
    return file_dataset,datas


def prepare_ad_event_based_dataset(file_dataset,truth_dataset=None,tot_evt=None,shuffle=True):
    if tot_evt!=None and tot_evt!=-1:
        file_dataset = file_dataset[:int(tot_evt),:,]
        truth_dataset = truth_dataset[:int(tot_evt)]

    #truth_dataset =  np.ones(file_dataset[:,:,0].shape[0])
    #if truth_name!=None:
    #    truth_dataset.fill(inverse_dict_map(process_name_dict)[truth_name])
        
    particle_id = file_dataset[:,:,[3]]
    particle_id = np.where(particle_id>1,particle_id-1,particle_id)
    phi_sin = np.sin(file_dataset[:,:,[2]])
    #phi_cos = np.cos(file_dataset[:,:,[2]])
    phi_cos = np.where(file_dataset[:,:,[2]]!=0,np.cos(file_dataset[:,:,[2]]),0)

    #have to concatenate truth dataset to the file_dataset
    file_dataset = np.concatenate([np.repeat(np.expand_dims(np.expand_dims(truth_dataset,axis=-1),axis=-1),
                                    file_dataset.shape[1],axis=1),
                                    particle_id, #pid
                                    file_dataset[:,:,0:2], #pt,eta
                                    phi_cos,
                                    phi_sin],
                                    axis=-1)
    if shuffle:
        random.shuffle(file_dataset)
    return file_dataset




def prepare_ad_event_based_h5file(outfile,true_labels,input_feats_per_batch, input_feats_met,pred_feats_per_batch,pred_feats_met,loss_dict):

    #update features pid to increase them by 1 for everything that is non zero
    input_pid = input_feats_per_batch[:,:,[0]]
    pred_pid = pred_feats_per_batch[:,:,[0]]
    input_pid = np.where(input_pid>0,input_pid+1,input_pid)
    pred_pid = np.where(pred_pid>0,pred_pid+1,pred_pid)

    pred_features_upd = np.concatenate([pred_pid,pred_feats_per_batch[:,:,1:]],axis=-1)
    input_features_upd = np.concatenate([input_pid,input_feats_per_batch[:,:,1:]],axis=-1)

    #insert 0 for eta and 1 for pid for met
    input_met_expanded =  np.expand_dims(np.insert(input_feats_met, [0,1], [1,0], axis=1),axis=1)
    pred_met_expanded =  np.expand_dims(np.insert(pred_feats_met, [0,1], [1,0], axis=1),axis=1)


    #print(input_met_expanded.shape,input_pid.shape,input_feats_per_batch[:,:,1:].shape)

    file_dataset_input = np.concatenate([input_met_expanded,input_features_upd],axis=1)
    file_dataset_pred = np.concatenate([pred_met_expanded,pred_features_upd],axis=1)

    with h5py.File(outfile, 'w') as handle:
        handle.create_dataset('InputParticles', data=file_dataset_input, compression='gzip')
        handle.create_dataset('PredictedParticles', data=file_dataset_pred, compression='gzip')
        handle.create_dataset('ProcessID', data=true_labels, compression='gzip')
        for key in loss_dict.keys():
            loss_label = 'Loss_'+key
            handle.create_dataset(loss_label, data=loss_dict[key], compression='gzip')


process_name_dict = {0: 'WW_13TeV_50PU',
                    1: 'WZ_13TeV_50PU',
                    2: 'ZZ_13TeV_50PU',
                    3: 'qcd_13TeV_50PU',
                    4: 'ttbar_13TeV_50PU',
                    5: 'Zjets_13TeV_50PU',
                    6: 'gammajets_13TeV_50PU',
                    7: 'Wjets_13TeV_50PU',
                    30: 'Ato4l',
                    31: 'hChToTauNu',
                    32: 'hToTauTau',
                    33: 'LQ'
                    }
                    
def inverse_dict_map(f):
    return f.__class__(map(reversed, f.items()))


def reshape_to_dense_batch(features,batch_size):
    s0 = features.shape[0]
    s1 = features.shape[1]
    s2 = features.shape[2]
    features = features.reshape((int(s0*batch_size),int(s1/batch_size),s2))
    return features


def prepare_final_output_features(pred_features,pred_features_met,num_classes,batch_size):
    pred_id = np.expand_dims(np.argmax(np.exp(pred_features[:,:,0:num_classes]),axis=-1),axis=-1)
    pred_features_batch = np.concatenate([pred_id,pred_features[:,:,num_classes:]],axis=-1)
    pred_features_merged = pred_features_batch.reshape(-1,pred_features_batch.shape[2])
    return pred_features_merged, pred_features_batch, pred_features_met


def prepare_final_input_features(prepared_dataset,batch_size):
    len_drop_last = batch_size*(len(prepared_dataset)//batch_size)
    t_per_batch = prepared_dataset[:len_drop_last,1:,1:]
    t = t_per_batch.reshape(-1,t_per_batch.shape[-1])
    t_met = prepared_dataset[:len_drop_last,0:1,[2,4,5]]
    t_met = t_met.reshape((t_met.shape[0])*t_met.shape[1],t_met.shape[2])
    true_labels = prepared_dataset[:len_drop_last,0,[0]]
    return t, t_per_batch, t_met, true_labels
        
