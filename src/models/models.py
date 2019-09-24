import torch
import torch.nn as nn
from .layers import GConv, EdgeCompute, Block

import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F

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
        h = self.activation(h)
        return {'h' : h}  


class GCN(nn.Module):
    """Define a GCN layer"""
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(2*in_feats, out_feats, activation)
    
    def forward(self, g):
        # Initialize the node features with h.
        g.update_all(message_func, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g
  

class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GNN, self).__init__()
        self.layers = nn.ModuleList([
            GCN(in_dim, hidden_dim, F.relu),
            GCN(hidden_dim, out_dim, F.relu)])
    
    def forward(self, g):
        # For undirected graphs, in_degree is the same as
        # out_degree.
    
        for conv in self.layers:
            g = conv(g)

        return g 

