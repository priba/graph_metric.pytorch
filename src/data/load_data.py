# -*- coding: utf-8 -*-
from __future__ import print_function, division

"""
Load the corresponding dataset.
"""

import torch
import numpy as np
import glob
from torch.utils.data import DataLoader
from . import data_utils as du
import os
import glob
import re
from tqdm import tqdm
import pickle

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"


def load_data(dataset, data_path, triplet=False, batch_size=32, prefetch=4):
    if dataset == 'letters':
        data_train, data_valid, data_test, gallery = load_letters(data_path, triplet)
        train_loader = DataLoader(data_train, batch_size=batch_size, num_workers=prefetch, collate_fn=du.collate_fn_multiple_size_siamese, shuffle=True)

        if triplet:
            batch_size = 3*batch_size
        else:
            batch_size = 2*batch_size

        # Load same numbers of graphs that are asked in training
        valid_loader = DataLoader(data_valid, batch_size=1, collate_fn=du.collate_fn_multiple_size)
        test_loader = DataLoader(data_test, batch_size=1, collate_fn=du.collate_fn_multiple_size)
        gallery_loader = DataLoader(gallery, batch_size=batch_size, collate_fn=du.collate_fn_multiple_size, num_workers=prefetch)
        node_size=2
        return train_loader, valid_loader, test_loader, gallery_loader, node_size
    elif dataset == 'histograph':
        sys.exit()
        return load_histograph(data_path)
    raise NameError(dataset + ' not implemented!')


def load_letters(data_path, triplet=False):
    from .Letters import Letters_train, Letters, create_graph_letter

    pickle_dir = os.path.join(data_path, os.pardir, os.pardir, 'Letters-pickled', os.path.basename(data_path))
    if not os.path.isdir(pickle_dir):
        # Data to pickle
        dataset_to_pickle(data_path, pickle_dir, create_graph_letter, '.gxl')
    
    # Get data for train, validation and test
    data_train = Letters_train(pickle_dir, os.path.join(data_path, 'train.cxl'), triplet)
    data_valid = Letters(pickle_dir, os.path.join(data_path, 'validation.cxl'))
    data_test = Letters(pickle_dir, os.path.join(data_path, 'test.cxl'))
    gallery = Letters(pickle_dir, os.path.join(data_path, 'train.cxl'))
    return data_train, data_valid, data_test, gallery


def dataset_to_pickle(root_path, out_path, graph_reader, graph_ext):
    file_list = glob.glob(os.path.join(root_path, '*'+graph_ext))
    id_list = np.array([re.search(os.path.join(r'^'+root_path, '(.*)'+graph_ext+'$'), s).group(1) for s in file_list])
    os.makedirs(out_path)
    for f in tqdm(id_list):
        node_labels, am = graph_reader(os.path.join(root_path, f + graph_ext))
        graph_dict = {'node_labels': node_labels, 'am': am}
        pickle_file = os.path.join(out_path, f+'.p')
        pickle.dump(graph_dict, open(pickle_file, "wb"))

def load_histographi(data_path, representation='adj'):
    data_train = datasets.HistoGraph(data_path, '../../../02_GXL/02_PAR/01_Keypoint/2/', 'train.txt', representation, normalization)
    data_valid = datasets.HistoGraph(data_path, '../../../02_GXL/02_PAR/01_Keypoint/2/', 'valid.txt', representation, normalization)
    data_test = datasets.HistoGraph(data_path, '../../../02_GXL/02_PAR/01_Keypoint/2/', 'test.txt', representation, normalization)
    return data_train, data_valid, data_test

