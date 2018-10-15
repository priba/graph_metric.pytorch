#-*- coding: utf-8 -*-
from __future__ import print_function, division

"""
Pytorch useful tools.
"""

import torch
import os
import errno
import numpy as np
from sklearn.metrics import average_precision_score
from joblib import Parallel, delayed
import multiprocessing

__author__ = 'Pau Riba'
__email__ = 'priba@cvc.uab.cat'


# Data
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


def graph_cat(g):
    nodes = []
    indices = []
    data = []
    g_size = []
    offset = 0
    for gi in g:
        nodes.append(gi[0])
        indices.append(gi[1]._indices()+offset)
        data.append(gi[1]._values())
        g_size.append(gi[2])
        offset = offset + gi[0].size(0)
    g_out = (
            torch.cat(nodes),
            torch.cat(indices, dim=1),
            torch.cat(data),
            torch.cat(g_size)
            )
    return graph_to_sparse(g_out)


# Evaluation
def knn_accuracy(dist_matrix, target_gallery, target_query, k=5):
    # Predict
    ind = np.argsort(dist_matrix, axis=1)
    sort_target = target_gallery[ind]
    sort_target = sort_target[:,:k]
    
    # Counts
    counts = np.zeros(sort_target.shape)
    for i in range(k):
        counts[:,i] = (sort_target[:, i].unsqueeze(1) == sort_target).long().sum(1)

    predict_ind = counts.argmax(axis=1)
    predict = [sort_target[i, pi] for i, pi in enumerate(predict_ind)]
    predict = torch.stack(predict)

    # Accuracy
    acc = predict.eq(target_query).float().sum()
    acc = 100.0*acc/predict.size(0)
    return acc


def mean_average_precision(dist_matrix, target_gallery, target_query):
    # Number of queries
    nq = target_query.size(0)

    # Distance to similarity
    sim = 1./(1+dist_matrix)

    # Relevant items
    str_sim = target_query.unsqueeze(1) == target_gallery
    if sim.is_cuda:
        sim, str_sim = sim.cpu(), str_sim.cpu()
    sim, str_sim = sim.numpy(), str_sim.numpy()

    num_cores = min(multiprocessing.cpu_count(), 32)
    aps = Parallel(n_jobs=num_cores)(delayed(average_precision_score)(str_sim[iq], sim[iq]) for iq in range(nq))
    # If str_sim is all 0, aps is nan
    ind = [i for i, ap in enumerate(aps) if np.isnan(ap)]
    for i in ind:
        aps[i] = 0.0
    return np.mean(aps) 
    
# Checkpoints
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

