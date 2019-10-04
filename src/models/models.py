import torch
import torch.nn as nn
from .layers import GConv, EdgeCompute, Block

import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GATConv

# Sends a message of node feature h.
def message_func(edges):
    return {'m': edges.src['h']}



def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.sum(nodes.mailbox['m'], 1)
    return {'m': accum}


class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = torch.cat([node.data['h'], node.data['m']],1)
        h = self.linear(h)
        if self.activation is not None:
            h = self.activation(h)
        return {'h' : h}  


class GCN(nn.Module):
    """Define a GCN layer"""
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(2*in_feats, out_feats, activation)
    
    def forward(self, g):
        # Initialize the node features with h.
        g.apply_edges(edge_attention)
        g.update_all(message_func, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g
  

class GNN_old(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GNN, self).__init__()
        self.layers = nn.ModuleList([
            GCN(in_dim, hidden_dim, F.relu),
            GCN(hidden_dim, out_dim, None)])
    
    def forward(self, g):
        # For undirected graphs, in_degree is the same as
        # out_degree.
         
        for conv in self.layers:
            g = conv(g)

        return g 

class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=4): 
        super(GNN, self).__init__()
        self.layers = nn.ModuleList([
            GATConv(in_dim, hidden_dim, heads, residual=True, activation=F.relu),
            GATConv(heads*hidden_dim, hidden_dim, heads, feat_drop=0.3, residual=True, activation=F.relu)])

        self.last_layer = GATConv(heads*hidden_dim, out_dim, heads, residual=True)

    def forward(self, g):
        h = g.ndata['h']


        for conv in self.layers:
            h = conv(g, h)
            h = h.view(h.shape[0], -1)

        h = self.last_layer(g, h)
        h = h.mean(1)

        g.ndata['h'] = h
        return g

