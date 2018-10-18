# -*- coding: utf-8 -*-
from __future__ import print_function, division

"""
Contrastive loss function.

Based on: https://github.com/delijati/pytorch-siamese/blob/master/contrastive.py
"""

import torch
import torch.nn as nn
from .distance import SoftHd

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0, reduction='elementwise_mean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.distance = SoftHd()

    def forward(self, g1, g2, y):
        d = self.distance(g1, g2, mode='pairs')

        md = self.margin - d
        md = torch.clamp(md, min=0.0)
        loss = y * md + (1 - y) * d
        
        if self.reduction == 'none':
            return loss
        
        loss = loss.sum()
        if self.reduction=='sum':
            return loss
        elif self.reduction=='elementwise_mean':
            loss = loss / g1[0].size(0)
            return loss
        
        raise NameError(self.reduction + ' not implemented!')

    def getDistance(self):
        return self.distance

class TripletLoss(nn.Module):

    def __init__(self, margin=1.0, swap=False, reduction='elementwise_mean',dist=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.swap = swap
        self.reduction = reduction
        self.distance = SoftHd()
        self.dist = dist

    def forward(self, anc, pos, neg):
        d_pos = self.distance(anc, pos, mode='pairs')
        d_neg = self.distance(anc, neg, mode='pairs')
        if self.swap:
            d_neg_aux = self.distance(pos, neg, mode='pairs')
            d_neg = torch.min(d_neg, d_neg_aux)

        loss = torch.clamp(d_pos-d_neg+self.margin, 0.0)
        if self.dist:
            loss = loss + d_pos
        if self.reduction == 'none':
            return loss
        
        loss = loss.sum()
        if self.reduction=='sum':
            return loss
        elif self.reduction=='elementwise_mean':
            loss = loss / anc[0].size(0)
            return loss
        
        raise NameError(self.reduction + ' not implemented!')

    def getDistance(self):
        return self.distance
