#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Graph Distance
"""

import torch
import torch.nn as nn
from fml.functional import pairwise_distances, sinkhorn

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"

class SoftHd(nn.Module):
    def __init__(self, in_sz):
        super(SoftHd, self).__init__()
        self.ins_del_cost = nn.Parameter(torch.FloatTensor([2]))

    def cdist(self, set1, set2):
        ''' Pairwise Distance between two matrices
        Input:  x is a Nxd matrix
                y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
        Source: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        '''

        xx = set1.unsqueeze(1).expand((set1.size(0), set2.size(0), set1.size(1)))
        yy = set2.unsqueeze(0).expand_as(xx)

        return (xx - yy).pow(2).sum(-1)


    def soft_hausdorff(self, set1, connections1, set2, connections2):
        dist_matrix = self.cdist(set1, set2)

        # Insertions and delitions
        dist_matrix = torch.cat([dist_matrix, (self.ins_del_cost*connections1).unsqueeze(1)], dim=1)
        zero_diagonal = torch.zeros(1)
        if dist_matrix.is_cuda:
            zero_diagonal = zero_diagonal.cuda()
        dist_matrix = torch.cat([dist_matrix, torch.cat([self.ins_del_cost*connections2, zero_diagonal]).unsqueeze(0)], dim=0)
        # \sum_{a\in set1} \inf_{b_\in set2} d(a,b)
        a, indA = dist_matrix.min(0)
        a = a.sum()
         
        # \sum_{b\in set2} \inf_{a_\in set1} d(a,b)
        b, indB = dist_matrix.min(1)
        b = b.sum()
        d = a + b
        d = d/2.0
        # d = d/min(dist_matrix.shape)
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


class Wasserstein(nn.Module):
    def __init__(self):
        super(Wasserstein, self).__init__()
        self.ins_del_cost = nn.Parameter(torch.FloatTensor([2]))

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

        set1 = torch.zeros((sz2.size(0), sz1.max(), x1.size(-1)), device=x1.device)
        set2 = torch.zeros((sz2.size(0), sz2.max(), x2.size(-1)), device=x2.device)
        a = torch.ones(set1.shape[0:2],
                       requires_grad=False,
                       device=set1.device)

        b = torch.ones(set2.shape[0:2],
                       requires_grad=False,
                       device=set2.device)

        for i in range(bz):
            if mode == 'retrieval':
                set1[i] = x1
            else:
                set1[i,:sz1[i]] = x1[start1:start1+sz1[i]]
                a[i,sz1[i]:] = 0
                start1 = start1 + sz1[i]

            set2[i,:sz2[i]] = x2[start2:start2+sz2[i]]
            b[i,sz2[i]:] = 0

            start2 = start2 + sz2[i]

        b = b * a.sum(1, keepdim=True) / b.sum(1, keepdim=True)
        if mode == 'retrieval':
            steps = [i for i in range(0, bz, 128)] + [bz]
            d = []
            for i in range(len(steps)-1):
                M = pairwise_distances(set1[steps[i]:steps[i+1]], set2[steps[i]:steps[i+1]])
                P = sinkhorn(a[steps[i]:steps[i+1]], b[steps[i]:steps[i+1]], M, 1e-3, max_iters=500, stop_thresh=1e-3)
                d.append((M * P).sum((1,2)))
            d = torch.cat(d)
        else:
            M = pairwise_distances(set1, set2, p=2)
            P = sinkhorn(a, b, M.detach(), 1e-3, max_iters=500, stop_thresh=1e-1)
            d = (M * P).sum((1,2))
        return d

