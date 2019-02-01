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
    if g[1].shape[0]==0:
        g_out = ( g[0],
                  torch.sparse.FloatTensor(size=(g[0].shape[0], g[0].shape[0])),
                  g[3]
                )
    else:
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
    _, ind = dist_matrix.sort(1)
    sort_target = target_gallery[ind.cpu()]
    sort_target = sort_target[:,:k]
    
    # Counts
    counts = np.zeros(sort_target.shape)
    for i in range(k):
        #counts[:,i] = (sort_target[:, i].unsqueeze(1) == sort_target).long().sum(1)
        counts[:,i] = (np.expand_dims(sort_target[:, i], axis=1) == sort_target).sum(1)
    
    predict_ind = counts.argmax(1)
    predict = [sort_target[i, pi] for i, pi in enumerate(predict_ind)]
    predict = np.stack(predict)

    # Accuracy
    acc = (predict == target_query).astype(np.float).sum()
    acc = 100.0*acc/predict.shape[0]
    return acc


def mean_average_precision(dist_matrix, target_gallery, target_query):
    # Number of queries
    nq = target_query.shape[0]

    # Distance to similarity
    sim = 1./(1+dist_matrix)

    # Relevant items
    str_sim = (np.expand_dims(target_query, axis=1) == np.expand_dims(target_gallery, axis=0)) * 1

    num_cores = min(multiprocessing.cpu_count(), 32)
    aps = Parallel(n_jobs=num_cores)(delayed(average_precision_score)(str_sim[iq], sim[iq].cpu().numpy()) for iq in range(nq))
    # If str_sim is all 0, aps is nan
    ind = [i for i, ap in enumerate(aps) if np.isnan(ap)]
    for i in sorted(ind, reverse=True):
        del aps[i]
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
        print("=> loaded model '{}' (epoch {}, Best Performance {})".format(model_file, checkpoint['epoch'], checkpoint['best_perf']))
        return checkpoint
    else:
        print("=> no model found at '{}'".format(model_file))
        raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), model_file)

