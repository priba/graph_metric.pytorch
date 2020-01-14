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


# Evaluation
def knn_accuracy(dist_matrix, target_gallery, target_query, k=5, dataset='gw'):
    # Predict
    _, ind = dist_matrix.sort(1)
    sort_target = target_gallery[ind.cpu()]
    if  True: #dataset=='gw':
        sort_target = sort_target[:,:k]
    else:
        sort_target = sort_target[:,1:k+1]

    # Counts
    counts = np.zeros(sort_target.shape)
    for i in range(k):
        counts[:,i] = (np.expand_dims(sort_target[:, i], axis=1) == sort_target).sum(1)

    predict_ind = counts.argmax(1)
    predict = [sort_target[i, pi] for i, pi in enumerate(predict_ind)]
    predict = np.stack(predict)

    # Accuracy
    acc = (predict == target_query).astype(np.float).sum()
    acc = 100.0*acc/predict.shape[0]
    return acc


def mean_average_precision(dist_matrix, target_gallery, target_query, dataset='gw'):
    # Number of queries
    nq = target_query.shape[0]

    # Distance to similarity
    sim = 1/(1+dist_matrix)

    # Relevant items
    str_sim = (np.expand_dims(target_query, axis=1) == np.expand_dims(target_gallery, axis=0)) * 1

    # Self comparison
    if False: #dataset=='ak':
        id_x, id_y = torch.where(sim==1)
        new_sim = torch.zeros(sim.shape[0], sim.shape[1]-1).to(sim.device)
        new_str_sim = np.zeros([sim.shape[0], sim.shape[1]-1])
        for i in range(id_x.shape[0]):
            new_sim[i] = torch.cat([sim[i,:id_y[i]], sim[i,id_y[i]+1:]])
            new_str_sim[i] = np.concatenate([str_sim[i,:id_y[i]], str_sim[i,id_y[i]+1:]])
        sim = new_sim
        str_sim = new_str_sim

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

