import torch
import torch.nn as nn

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GATConv
from .layers import GatedGraphConv, EdgeConv


class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=4, dropout=0.3):
        super(GNN, self).__init__()

        self.embedding = nn.Linear(in_dim, hidden_dim)
        self.edge_embedding = EdgeConv(hidden_dim, hidden_dim, activation=torch.relu)
        self.edge_func = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(True), nn.Linear(128, hidden_dim*hidden_dim), nn.Tanh())
        self.layers = GatedGraphConv(hidden_dim, hidden_dim, 3, self.edge_func, 1, dropout=dropout)
        self.last_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, g):
        h = g.ndata['h']

        h = self.embedding(h)
        he = self.edge_embedding(g, h)
        h = self.layers(g, h, he)
        h = self.last_layer(h)

        h = torch.tanh(h)
        g.ndata['h'] = h

        return g

