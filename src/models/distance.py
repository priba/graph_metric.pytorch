#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Graph Distance
"""

import torch
import torch.nn as nn

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"

class SoftHd(nn.Module):
    def __init__(self, in_sz):
        super(SoftHd, self).__init__()
        self.head = nn.Linear(in_sz, 1)
        self.ins_del_cost = nn.Parameter(torch.FloatTensor([5]))

    def cdist(self, set1, set2, p=1.0):
        ''' Pairwise Distance between two matrices
        Input:  x is a Nxd matrix
                y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
        Source: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        '''
#        x_norm = (set1**2).sum(1).view(-1, 1)
#        y_t = set2.t()
#        y_norm = (set2**2).sum(1).view(1, -1)
        
#        dist_matrix = x_norm + y_norm - 2.0 * torch.mm(set1, y_t)
#        return dist_matrix
        xx = set1.unsqueeze(1).expand((set1.size(0), set2.size(0), set1.size(1)))
        yy = set2.unsqueeze(0).expand_as(xx)
#        dxy = self.head((xx-yy).abs()).squeeze()
        return (xx - yy).abs().pow(p).sum(-1)


    def soft_hausdorff(self, set1, connections1, set2, connections2):
        dist_matrix = self.cdist(set1, set2, p=2)

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
        x1, am1, sz1 = g1
        x2, am2, sz2 = g2
        
        
        oneVec1 = torch.ones(am1.shape[1]).unsqueeze(1)
        oneVec2 = torch.ones(am2.shape[1]).unsqueeze(1)
        if x1.is_cuda:
            oneVec1, oneVec2 = oneVec1.cuda(), oneVec2.cuda()
        
        if am1._nnz() == 0:
            conn1 = oneVec1
        else:
            conn1 = torch.sparse.mm(am1, oneVec1) + oneVec1

        if am2._nnz() == 0:
            conn2 = oneVec2
        else:
            conn2 = torch.sparse.mm(am2, oneVec2) + oneVec2

        conn1, conn2 = conn1.squeeze(), conn2.squeeze()

        bz = sz2.shape[0] # Batch Size

        d = torch.zeros(bz)
        if x1.is_cuda:
            d = d.cuda()

        start1 = 0
        start2 = 0
        for i in range(bz):
            if mode == 'pairs':
                d[i],_,_ = self.soft_hausdorff(x1[start1:start1+sz1[i]], conn1[start1:start1+sz1[i]], x2[start2:start2+sz2[i]], conn2[start2:start2+sz2[i]])
                start1 = start1 + sz1[i]
            elif mode == 'retrieval':
                d[i],_,_ = self.soft_hausdorff(x1, conn1, x2[start2:start2+sz2[i]], conn2[start2:start2+sz2[i]])
            else:
                raise NameError(mode + ' not implemented!')
            start2 = start2 + sz2[i]
        return d

