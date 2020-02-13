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
    sort_target = sort_target[:,:k]

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


def mean_average_precision(dist_matrix, target_gallery, target_query):
    # Number of queries
    nq = target_query.shape[0]

    aps = []
    for q in range(nq):
        _, indices = dist_matrix[q].sort()
        rel = np.array(target_query[q] == target_gallery[indices.cpu()])
        x=np.float32(np.cumsum(rel))/range(1,len(rel)+1)
        aps.append( np.sum(x[rel])/(len(x[rel])+10**-7))

    return np.mean(aps)


# def mean_average_precision(dist_matrix, target_gallery, target_query):
#     # Number of queries
#     nq = target_query.shape[0]
#
#     interpolation_points = np.linspace(0,1,11)
#
#     aps = []
#     for q in range(nq):
#         _, indices = dist_matrix[q].sort()
#         rel = np.array(target_query[q] == target_gallery[indices.cpu()])
#
#         recall = np.float32(np.cumsum(rel))/rel.sum()
#         precision = np.float32(np.cumsum(rel))/range(1,len(rel)+1)
#
#         prec = [precision[recall>=i].max() for i in interpolation_points]
#         aps.append( np.mean(prec))
#
#     return np.mean(aps)

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

