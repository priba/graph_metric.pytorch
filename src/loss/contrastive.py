# -*- coding: utf-8 -*-
from __future__ import print_function, division

"""
Contrastive loss function.

Based on: https://github.com/delijati/pytorch-siamese/blob/master/contrastive.py
"""

import torch
import torch.nn as nn
import .distance
__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0, reduction='elementwise_mean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, g1, g2, y):
        d = dist(g1, g2)

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


class TripletLoss(nn.Module):

    def __init__(self, margin=1.0, swap=False, reduction='elementwise_mean'):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.swap = swap
        self.reduction = reduction

    def forward(self, anc, pos, neg):
        d_pos = dist(anc, pos)
        d_neg = dist(anc, neg)
        if self.swap.
            d_neg_aux = dist(pos, neg)
            d_neg = torch.max(d_neg, d_neg_aux)

        loss = torch.max(0, d_pos-d_neg+self.margin)

        if self.reduction == 'none':
            return loss
        
        loss = loss.sum()
        if self.reduction=='sum':
            return loss
        elif self.reduction=='elementwise_mean':
            loss = loss / anc[0].size(0)
            return loss
        
        raise NameError(self.reduction + ' not implemented!')

