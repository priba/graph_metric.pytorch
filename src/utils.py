#-*- coding: utf-8 -*-
from __future__ import print_function, division

"""
Pytorch useful tools.
"""

import torch
import os
import errno
import numpy as np

__author__ = 'Pau Riba'
__email__ = 'priba@cvc.uab.cat'


def graph_to_sparse(g):
    g_out = ( g[0],
              torch.sparse.FloatTensor(g[1], g[2], (g[0].shape[0], g[0].shape[0])),
              g[3]
            )
    return g_out


def graph_cuda(g):
    """ Graph is represented as
    g (nodes, indices, data, target)
    output (nodes, am [sparse], target)
    """
    g = tuple((gi.cuda() for gi in g) )
    return g


def save_checkpoint(state, directory, file_name):

    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, file_name + '.pth')
    torch.save(state, checkpoint_file)


def load_checkpoint(model_file):
    if os.path.isfile(model_file):
        print("=> loading model '{}'".format(model_file))
        checkpoint = torch.load(model_file)
        print("=> loaded model '{}' (epoch {}, acc {})".format(model_file, checkpoint['epoch'], checkpoint['best_acc']))
        return checkpoint
    else:
        print("=> no model found at '{}'".format(model_file))
        raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), model_file)

