#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Graph Distance
"""

import torch
import torch.nn as nn

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"

class SoftHd(nn.Module):
    def __init__(self):
        super(SoftHd, self).__init__()


    def cdist(self, set1, set2):
        ''' Pairwise Distance between tow matrices
        Input:  x is a Nxd matrix
                y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
        Source: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        '''
        x_norm = (set1**2).sum(1).view(-1, 1)
        y_t = set2.t()
        y_norm = (set2**2).sum(1).view(1, -1)
        
        dist_matrix = x_norm + y_norm - 2.0 * torch.mm(set1, y_t)
        
        return dist_matrix

    def soft_hausdorff(self, set1, set2):
        dist_matrix = self.cdist(set1, set2)

        # \sum_{a\in set1} \inf_{b_\in set2} d(a,b)
        a = dist_matrix.min(0)[0].sum()
        
        # \sum_{b\in set2} \inf_{a_\in set1} d(a,b)
        b = dist_matrix.min(1)[0].sum()

        d = a + b
        d = d/(set1.shape[0] + set2.shape[0])
        return d


    def forward(self, g1, g2):
        x1, _, sz1 = g1
        x2, _, sz2 = g2

        bz = sz1.shape[0] # Batch Size

        d = torch.zeros(bz)
        if x1.is_cuda:
            d = d.cuda()

        start1 = 0
        start2 = 0

        for i in range(bz):
            d[i] = self.soft_hausdorff(x1[start1:start1+sz1[i]], x2[start2:start2+sz2[i]])
        
        return d
