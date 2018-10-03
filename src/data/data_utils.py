#-*- coding: utf-8 -*-
from __future__ import print_function, division

import numpy as np
import torch
from scipy.sparse import coo_matrix
from . import data_utils as du

"""
    Useful data tools.
"""

__author__ = 'Pau Riba'
__email__ = 'priba@cvc.uab.cat'


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor.
    From: https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py
    """
    import torch
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def collate_fn_multiple_size(batch):
    # Batch is a list where each element
    #   * Nodes (VxF)
    #   * Edges:(rows, columns, data)
    n_batch = len(batch)

    # Get the size of each graph in terms of number of nodes
    g_size = torch.LongTensor([x[0][0].shape[0] for x in batch])
    nl = []
    row = []
    col = []
    data = []
    targets = torch.LongTensor([x[-1] for x in batch])
    
    offset = 0

    for i in range(n_batch):
        nl.append(batch[i][0][0])
        row.append(batch[i][0][1][0]+offset)
        col.append(batch[i][0][1][1]+offset)
        data.append(batch[i][0][1][2])
        offset += g_size[i]

    nl = torch.FloatTensor(np.concatenate(nl, axis=0))
    row = np.concatenate(row, axis=0)
    col = np.concatenate(col, axis=0)
    data = np.concatenate(data, axis=0)
    am = coo_matrix((data, (row, col)), shape=(g_size.sum(), g_size.sum()) )
    am = du.sparse_mx_to_torch_sparse_tensor(am)
    
    return (nl, am, g_size), targets


def collate_fn_multiple_size_siamese(batch):
    # Batch is a list where each element
    #   * Nodes (VxF)
    #   * Edges:(rows, columns, data)
    n_batch = len(batch)

    # Check if we are in triplet setting
    triplet = False
    if batch[0][3].size()[0] == 0:
        triplet = True

    # Get the size of each graph in terms of number of nodes
    g_size1 = torch.LongTensor([x[0][0].shape[0] for x in batch])
    nl1 = []
    row1 = []
    col1 = []
    data1 = []

    g_size2 = torch.LongTensor([x[1][0].shape[0] for x in batch])
    nl2 = []
    row2 = []
    col2 = []
    data2 = []

    if triplet:
        g_size3 = torch.LongTensor([x[2][0].shape[0] for x in batch])
        nl3 = []
        row3 = []
        col3 = []
        data3 = []
        targets = None
    else: 
        g_size3 = None
        nl3 = None
        row3 = None
        col3 = None
        data3 = None
        targets = torch.cat([x[-1] for x in batch])
    
    offset1 = 0
    offset2 = 0
    if triplet:
        offset3 = 0

    for i in range(n_batch):
        # G1
        nl1.append(batch[i][0][0])
        row1.append(batch[i][0][1][0]+offset1)
        col1.append(batch[i][0][1][1]+offset1)
        data1.append(batch[i][0][1][2])
        offset1 += g_size1[i]

        # G2
        nl2.append(batch[i][1][0])
        row2.append(batch[i][1][1][0]+offset2)
        col2.append(batch[i][1][1][1]+offset2)
        data2.append(batch[i][1][1][2])
        offset2 += g_size2[i]

        if triplet:
            # G3
            nl3.append(batch[i][2][0])
            row3.append(batch[i][2][1][0]+offset3)
            col3.append(batch[i][2][1][1]+offset3)
            data3.append(batch[i][2][1][2])
            offset3 += g_size3[i]

    nl1 = torch.FloatTensor(np.concatenate(nl1, axis=0))
    row1 = np.concatenate(row1, axis=0)
    col1 = np.concatenate(col1, axis=0)
    data1 = np.concatenate(data1, axis=0)
    am1 = coo_matrix((data1, (row1, col1)), shape=(g_size1.sum(), g_size1.sum()) )
    am1 = du.sparse_mx_to_torch_sparse_tensor(am1)
    
    nl2 = torch.FloatTensor(np.concatenate(nl2, axis=0))
    row2 = np.concatenate(row2, axis=0)
    col2 = np.concatenate(col2, axis=0)
    data2 = np.concatenate(data2, axis=0)
    am2 = coo_matrix((data2, (row2, col2)), shape=(g_size2.sum(), g_size2.sum()) )
    am2 = du.sparse_mx_to_torch_sparse_tensor(am2)
    
    if triplet:
        nl3 = torch.FloatTensor(np.concatenate(nl3, axis=0))
        row3 = np.concatenate(row3, axis=0)
        col3 = np.concatenate(col3, axis=0)
        data3 = np.concatenate(data3, axis=0)
        am3 = coo_matrix((data3, (row3, col3)), shape=(g_size3.sum(), g_size3.sum()) )
        am3 = du.sparse_mx_to_torch_sparse_tensor(am3)
    
        return (nl1, am1, g_size1), (nl2, am2, g_size2), (nl3, am3, g_size3), targets
    return (nl1, am1, g_size1), (nl2, am2, g_size2), None, targets


