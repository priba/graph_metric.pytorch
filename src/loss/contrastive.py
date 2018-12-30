# -*- coding: utf-8 -*-
from __future__ import print_function, division

"""
Contrastive loss function.

Based on: https://github.com/delijati/pytorch-siamese/blob/master/contrastive.py
"""

import torch
import torch.nn as nn

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0, reduction='elementwise_mean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, g1, g2, y, distance):
        d = distance(g1, g2, mode='pairs')

        md = self.margin - d
        md = torch.clamp(md, min=0.0)
        loss = y * md + (1 - y) * d
        loss = loss/2.0
        
        if self.reduction == 'none':
            return loss
        
        loss = loss.sum()
        if self.reduction=='sum':
            return loss
        elif self.reduction=='elementwise_mean':
            loss = loss / y.size(0)
            return loss
        
        raise NameError(self.reduction + ' not implemented!')


class TripletLoss(nn.Module):

    def __init__(self, margin=1.0, swap=False, reduction='elementwise_mean',dist=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.swap = swap
        self.reduction = reduction
        self.dist = dist
        if self.dist:
            self.lambda_dist = 0.25

    def forward(self, anc, pos, neg, distance):
        d_pos = distance(anc, pos, mode='pairs')
        d_neg = distance(anc, neg, mode='pairs')
        if self.swap:
            d_neg_aux = distance(pos, neg, mode='pairs')
            d_neg = torch.min(d_neg, d_neg_aux)

        loss = torch.clamp(d_pos-d_neg+self.margin, 0.0)
        if self.dist:
            loss = loss + self.lambda_dist * d_pos
        if self.reduction == 'none':
            return loss
        
        loss = loss.sum()
        if self.reduction=='sum':
            return loss
        elif self.reduction=='elementwise_mean':
            loss = loss / d_pos.size(0)
            return loss
        
        raise NameError(self.reduction + ' not implemented!')

