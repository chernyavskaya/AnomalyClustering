import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, GCN, GCNConv, EdgeConv, GATConv, GATv2Conv, global_mean_pool, DynamicEdgeConv, BatchNorm


class EmbeddingLayer(nn.Module):
    """
    Embedding layer.
    Automatically splits input Tensors based on embedding sizes;
    then, embeds each feature separately and sums the embeddings
    back into a single outpuut Tensor.
    """

    def __init__(self, emb_szs,mode='sum'):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(in_sz, out_sz) for in_sz, out_sz in emb_szs])
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = [emb(x[..., i]) for i, emb in enumerate(self.embeddings)]
        if self.mode=='concat' :
            embedded_vars = torch.cat(x, dim=-1) #concatenate embedding , bad practice
        else :
            #We will sum embeddings instead 
            stacked_embedded_vars = torch.stack(x, dim=0)
            embedded_vars = torch.sum(stacked_embedded_vars, dim=0)
        return embedded_vars


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