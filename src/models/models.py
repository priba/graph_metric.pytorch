import torch
import torch.nn as nn

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GATConv


class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=4, dropout=0.3): 
        super(GNN, self).__init__()

        self.layers = nn.ModuleList([
            GATConv(in_dim, hidden_dim, heads, residual=True, activation=F.leaky_relu),
            GATConv(heads*hidden_dim, hidden_dim, heads, feat_drop=dropout, residual=True, activation=F.leaky_relu)])

        self.bn = nn.ModuleList([
            nn.BatchNorm1d(heads*hidden_dim),
            nn.BatchNorm1d(heads*hidden_dim)])

        self.last_layer = GATConv(heads*hidden_dim, out_dim, heads, residual=True)

    def forward(self, g):
        h = g.ndata['h']

        for i, conv in enumerate(self.layers):
            h = conv(g, h)
            h = h.view(h.shape[0], -1)
#            h = self.bn[i](h)

        h = self.last_layer(g, h)
        h = h.mean(1)

        h = torch.sigmoid(h)
        g.ndata['h'] = h

        return g

