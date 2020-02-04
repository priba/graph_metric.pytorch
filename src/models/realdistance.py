#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Graph Distance
"""

import torch
import dgl
import torch.nn as nn
import numpy as np

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"

class HausdorffEditDistance(nn.Module):
    def __init__(self, alpha=0.5, beta=0.1, tau_n=4., tau_e=16.):
        super(HausdorffEditDistance, self).__init__()
        self.register_buffer('alpha', torch.tensor([alpha]))
        self.register_buffer('beta', torch.tensor([beta, 1-beta]).unsqueeze(0).unsqueeze(0))
        self.register_buffer('tau_n', torch.tensor([tau_n]))
        self.register_buffer('tau_e', torch.tensor([tau_e]))
        self.p = 2

    def cdist(self, set1, set2, p=2.0):
        ''' Pairwise Distance between two matrices
        Input:  x is a Nxd matrix
                y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
        Source: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        '''
        xx = set1.unsqueeze(1).expand((set1.size(0), set2.size(0), set1.size(1)))
        yy = set2.unsqueeze(0).expand_as(xx)
        return (xx - yy).abs().pow(p)


    def soft_hausdorff(self, g1, g2, train=True):
        p1 = g1.ndata['pos']
        p2 = g2.ndata['pos']
        device = p1.device
        dtype = p1.dtype

        # Deletion
        d1_edges = g1.in_degrees().to(device).to(dtype)
        d1 = self.tau_n + d1_edges*self.tau_e/2.

        # Insertion
        d2_edges = g2.in_degrees().to(device).to(dtype)
        d2 = self.tau_n + d2_edges*self.tau_e/2.

        # Substitution
        beta = self.beta*g1.gdata['std']

        dist_matrix = self.cdist(p1, p2, p=2)
        dist_matrix = beta*dist_matrix
        dist_matrix = self.alpha*dist_matrix.sum(-1).sqrt()

        # Edges HED
        edges_hed = g1.in_degrees().unsqueeze(1)-g2.in_degrees().unsqueeze(0)
        edges_hed = self.tau_e*edges_hed.to(device).to(dtype).abs()

        dist_matrix = dist_matrix + edges_hed/2
        dist_matrix = dist_matrix/2.

        # \sum_{a\in set1} \inf_{b_\in set2} d(a,b)
        a, indA = dist_matrix.min(0)
        a = torch.min(a, d2)

        # \sum_{b\in set2} \inf_{a_\in set1} d(a,b)
        b, indB = dist_matrix.min(1)
        b = torch.min(b, d1)

        #d = a.mean() + b.mean()
        d = a.sum() + b.sum()

        upper_bound = (g1.number_of_nodes() - g2.number_of_nodes())*self.tau_n
        upper_bound = upper_bound.abs()
        if d < upper_bound:
            d = upper_bound.squeeze()

        normalization = (self.tau_n*(g1.number_of_nodes() + g2.number_of_nodes()) + self.tau_e*(g1.number_of_edges() + g2.number_of_edges()))
        d = d/normalization.squeeze()

        if train:
            return d

        indA[a==d2] = dist_matrix.shape[0]
        indB[b==d1] = dist_matrix.shape[1]
        return d, indB, indA


    def forward(self, g1, g2, mode='pairs'):
        ''' mode:   'pairs' expect paired graphs, same for g1 and g2.
                    'retrieval' g1 is just one graph and computes the distance against all graphs in g2
        '''

        g1_list = dgl.unbatch(g1)
        for i, g in enumerate(g1_list):
            g.gdata = {}
            g.gdata['std'] = g1.gdata['std'][i]

        g2_list = dgl.unbatch(g2)
        for i, g in enumerate(g2_list):
            g.gdata = {}
            g.gdata['std'] = g2.gdata['std'][i]

        d = []
        for i in range(len(g2_list)):
            if mode == 'pairs':
                d_aux = self.soft_hausdorff(g1_list[i], g2_list[i])
            elif mode == 'retrieval':
                query = g1_list[0]
                d_aux = self.soft_hausdorff(query, g2_list[i])
            else:
                raise NameError(mode + ' not implemented!')
            d.append(d_aux)
        d = torch.stack(d)
        return d

