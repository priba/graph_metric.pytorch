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
        self.head = nn.Linear(in_sz, 1)
        self.ins_del_cost = nn.Parameter(torch.FloatTensor([5]))
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
        return (xx - yy).abs().pow(p).sum(-1)


    def soft_hausdorff(self, g1, g2):
        dist_matrix = self.cdist(g1.ndata['h'], g2.ndata['h'], p=2)

        # Insertions and delitions
        #dist_matrix = torch.cat([dist_matrix, (self.ins_del_cost*connections1).unsqueeze(1)], dim=1)
        #zero_diagonal = torch.zeros(1)
        #if dist_matrix.is_cuda:
        #    zero_diagonal = zero_diagonal.cuda()
        #dist_matrix = torch.cat([dist_matrix, torch.cat([self.ins_del_cost*connections2, zero_diagonal]).unsqueeze(0)], dim=0)
        # \sum_{a\in set1} \inf_{b_\in set2} d(a,b)
        a, indA = dist_matrix.min(0)
        a = a.sum()
         
        # \sum_{b\in set2} \inf_{a_\in set1} d(a,b)
        b, indB = dist_matrix.min(1)
        b = b.sum()
        d = a + b
        d = d/min(dist_matrix.shape)
        return d, indB, indA


    def forward(self, g1, g2, mode='pairs'):
        ''' mode:   'pairs' expect paired graphs, same for g1 and g2.
                    'retrieval' g1 is just one graph and computes the distance against all graphs in g2
        '''
        g1_list = dgl.unbatch(g1)
        g2_list = dgl.unbatch(g2)

        d = []
        for i in range(len(g2_list)):
            if mode == 'pairs':
                d_aux,_,_ = self.soft_hausdorff(g1_list[i], g2_list[i])
            elif mode == 'retrieval':
                d_aux,_,_ = self.soft_hausdorff(g1_list[0], g2_list[i])
            else:
                raise NameError(mode + ' not implemented!')
            d.append(d_aux)
        d = torch.stack(d)
        return d

