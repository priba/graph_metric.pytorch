#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Graph Distance
"""

import torch
import torch.nn as nn

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"

class SoftHd(nn.Module):
    # Constructor
    def __init__(self, args={}):
        super(SoftHd, self).__init__()
        self.args = args

    def forward(self, g1, g2):
        d = 0
        return d
