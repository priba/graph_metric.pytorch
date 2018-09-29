# -*- coding: utf-8 -*-
from __future__ import print_function, division

"""
Load the corresponding dataset.
"""

import torch
import numpy as np
import glob

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"


def load_data(dataset, data_path, batch_size=32, num_workers=4, pin_memory=True):
    if dataset == 'letters':
        return load_letters(data_path)
    elif dataset == 'histograph':
        sys.exit()
        return load_histograph(data_path)
    raise NameError(dataset + ' not implemented!')


def load_letters(data_path, representation='adj'):
    from . import Letters
    import pdb; pdb.set_trace()
    # Get data for train, validation and test
    data_train = Letters(data_path, 'train.cxl')
    data_valid = Letters(data_path, 'validation.cxl')
    data_test = Letters(data_path, 'test.cxl')

    return data_train, data_valid, data_test


def load_histographi(data_path, representation='adj'):
    data_train = datasets.HistoGraph(data_path, '../../../02_GXL/02_PAR/01_Keypoint/2/', 'train.txt', representation, normalization)
    data_valid = datasets.HistoGraph(data_path, '../../../02_GXL/02_PAR/01_Keypoint/2/', 'valid.txt', representation, normalization)
    data_test = datasets.HistoGraph(data_path, '../../../02_GXL/02_PAR/01_Keypoint/2/', 'test.txt', representation, normalization)
    return data_train, data_valid, data_test


def collate_fn_multiple_size_siamese(batch):
    n_batch = len(batch)

    g_size1 = torch.LongTensor([x[0].size(0) for x in batch])
    g_size2 = torch.LongTensor([x[2].size(0) for x in batch])

    graph_size1 = torch.LongTensor([[x[0].size(0), x[0].size(1), x[1].size(2)] for x in batch])
    graph_size2 = torch.LongTensor([[x[2].size(0), x[2].size(1), x[3].size(2)] for x in batch])

    sz1, _ = graph_size1.max(dim=0)
    sz2, _ = graph_size2.max(dim=0)

    n_labels1 = torch.zeros(n_batch, sz1[0], sz1[1])
    n_labels2 = torch.zeros(n_batch, sz2[0], sz2[1])

    am1 = torch.zeros(n_batch, sz1[0], sz1[0], sz1[2])
    am2 = torch.zeros(n_batch, sz2[0], sz2[0], sz2[2])

    targets = torch.cat([x[-1] for x in batch])

    for i in range(n_batch):
        # Node Features
        n_labels1[i, :g_size1[i], :] = batch[i][0]
        n_labels2[i, :g_size2[i], :] = batch[i][2]

        # Adjacency matrix
        am1[i, :g_size1[i], :g_size1[i], :] = batch[i][1]
        am2[i, :g_size2[i], :g_size2[i], :] = batch[i][3]

    return n_labels1, am1, g_size1, n_labels2, am2, g_size2, targets

