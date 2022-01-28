import torch
import torch.nn as nn

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