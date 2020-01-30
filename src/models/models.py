import torch
import torch.nn as nn

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GATConv, GatedGraphConv
# from .layers import GatedGraphConv, EdgeConv


class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=4, dropout=0.3):
        super(GNN, self).__init__()

        self.embedding = nn.Linear(in_dim, hidden_dim)
#        self.edge_embedding = EdgeConv(hidden_dim, hidden_dim, activation=None)
#        self.edge_func = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(True), nn.Linear(128, hidden_dim*hidden_dim))
#        self.layers = GatedGraphConv(hidden_dim, hidden_dim, 3, self.edge_func, dropout=dropout)
        self.layers = GatedGraphConv(hidden_dim, hidden_dim, 3, 1)
        self.last_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, g):
        h = g.ndata['pos']

        h = self.embedding(h)
#        he = self.edge_embedding(g, h)
#        h = self.layers(g, h)
        h = self.layers(g, h, torch.zeros(g.edges()[0].shape[0]))
        h = self.last_layer(h)

        h = torch.tanh(h)
        g.ndata['h'] = h

        return g

