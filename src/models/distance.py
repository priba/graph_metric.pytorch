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

class SoftHd(nn.Module):
    def __init__(self, in_sz):
        super(SoftHd, self).__init__()
        self.node_ins_del_cost = nn.Sequential( nn.Linear(in_sz, 64),
                                           nn.ReLU(True),
                                           nn.Linear(64, 1))
#        self.edge_ins_del_cost = nn.Sequential( nn.Linear(in_sz, 64),
#                                           nn.ReLU(True),
#                                           nn.Linear(64, 1))
        self.p = 2

    def cdist(self, set1, set2):
        ''' Pairwise Distance between two matrices
        Input:  x is a Nxd matrix
                y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
        Source: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        '''
        dist = set1.unsqueeze(1) - set2.unsqueeze(0)
        return dist.abs()


    def soft_hausdorff(self, g1, g2, train=True):
#        if g1.number_of_nodes() < g2.number_of_nodes():
#            tmp = g2
#            g2 = g1
#            g1 = tmp

        h1 = g1.ndata['h']
        h2 = g2.ndata['h']

        p1 = g1.ndata['pos']
        p2 = g2.ndata['pos']

#        spatial_dist = self.cdist(p1, p2)
#        spatial_dist = spatial_dist.pow(2.).sum(-1).sqrt()

        feature_dist = self.cdist(h1, h2).pow(2.).sum(-1)/2
        dist_matrix = feature_dist

#        dist_matrix = spatial_dist + feature_dist

        d1 = self.node_ins_del_cost(h1).abs().squeeze()
        d2 = self.node_ins_del_cost(h2).abs().squeeze()

        # \sum_{a\in set1} \inf_{b_\in set2} d(a,b)
        a, indA = dist_matrix.min(0)
        a = torch.min(a, d2)

        # \sum_{b\in set2} \inf_{a_\in set1} d(a,b)
        b, indB = dist_matrix.min(1)
        b = torch.min(b, d1)

        #d = a.mean() + b.mean()
        d = a.sum() + b.sum()
        d = d/(d1.sum() + d2.sum()) # (a.shape[0] + b.shape[0])

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

