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
import _pickle as pickle
import sys

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"


def load_data(dataset, data_path, triplet=False, batch_size=32, prefetch=4, set_partition='cv1'):

    if dataset == 'iam':
        data_train, data_valid, data_test, gallery = load_iam(data_path, triplet)
        print_statistics(data_train, data_valid, gallery, data_test, gallery)
        train_loader = DataLoader(data_train, batch_size=batch_size, num_workers=prefetch, collate_fn=du.collate_fn_multiple_size_siamese, shuffle=True)

        if triplet:
            batch_size = 6*batch_size
        else:
            batch_size = 4*batch_size

        # Load same numbers of graphs that are asked in training
        valid_loader = DataLoader(data_valid, batch_size=1, collate_fn=du.collate_fn_multiple_size)
        test_loader = DataLoader(data_test, batch_size=1, collate_fn=du.collate_fn_multiple_size)
        gallery_loader = DataLoader(gallery, batch_size=batch_size, collate_fn=du.collate_fn_multiple_size, num_workers=prefetch)
        node_size=2
        return train_loader, valid_loader, gallery_loader, test_loader, gallery_loader, node_size
    elif dataset == 'histograph-gw':
        data_train, queries, gallery_valid, gallery_test = load_histograph_gw(data_path, triplet, set_partition=set_partition)
        print_statistics(data_train, queries, gallery_valid, queries, gallery_test)

        if triplet:
            anchors = np.array([data_train.labels[g[0]] for g in data_train.groups])
            anchors_counts = [(a==anchors).sum() for a in anchors]
            anchor_probs = 1/np.array(anchors_counts)
            train_sampler = torch.utils.data.sampler.WeightedRandomSampler(anchor_probs, 200000, replacement=True)
            train_loader = DataLoader(data_train, batch_size=batch_size, num_workers=prefetch, collate_fn=du.collate_fn_multiple_size_siamese, sampler=train_sampler)
            batch_size = 6*batch_size
        else:
            pairs = np.array([True if data_train.labels[g[0]]==data_train.labels[g[1]] else False for g in data_train.groups])
            weights = np.zeros(len(pairs))
            weights[pairs] = 1.0/(pairs+0.0).sum()
            weights[np.invert(pairs)] = 1.0/(np.invert(pairs)+0.0).sum()
            train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, 3000, replacement=True)
            train_loader = DataLoader(data_train, batch_size=batch_size, num_workers=prefetch, collate_fn=du.collate_fn_multiple_size_siamese, sampler=train_sampler, pin_memory=True)
            batch_size = 4*batch_size

        # Load same numbers of graphs that are asked in training
        queries_loader = DataLoader(queries, batch_size=1, collate_fn=du.collate_fn_multiple_size, shuffle=False)
        valid_gallery_loader = DataLoader(gallery_valid, batch_size=batch_size, collate_fn=du.collate_fn_multiple_size, num_workers=prefetch, shuffle=False)
        test_gallery_loader = DataLoader(gallery_test, batch_size=batch_size, collate_fn=du.collate_fn_multiple_size, num_workers=prefetch, shuffle=False)
        node_size=2
        return train_loader, queries_loader, valid_gallery_loader, queries_loader, test_gallery_loader, node_size
    elif dataset == 'histograph-ak':
        data_train, queries_valid, gallery_valid, queries_test, gallery_test = load_histograph_ak(data_path, triplet)
        print_statistics(data_train, queries_valid, gallery_valid, queries_test, gallery_test)

        if triplet:
            anchors = np.array([data_train.labels[g[0]] for g in data_train.groups])
            anchors_counts = [(a==anchors).sum() for a in anchors]
            anchor_probs = 1/np.array(anchors_counts)
            train_sampler = torch.utils.data.sampler.WeightedRandomSampler(anchor_probs, 100000, replacement=True)
            train_loader = DataLoader(data_train, batch_size=batch_size, num_workers=prefetch, collate_fn=du.collate_fn_multiple_size_siamese, sampler=train_sampler)
            batch_size = 6*batch_size
        else:
            pairs = np.array([True if data_train.labels[g[0]]==data_train.labels[g[1]] else False for g in data_train.groups])
            weights = np.zeros(len(pairs))
            weights[pairs] = 1.0/(pairs+0.0).sum()
            weights[np.invert(pairs)] = 1.0/(np.invert(pairs)+0.0).sum()
            train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, 3000, replacement=True)
            train_loader = DataLoader(data_train, batch_size=batch_size, num_workers=prefetch, collate_fn=du.collate_fn_multiple_size_siamese, sampler=train_sampler, pin_memory=True)
            batch_size = 4*batch_size

        # Load same numbers of graphs that are asked in training
        valid_queries_loader = DataLoader(queries_valid, batch_size=1, collate_fn=du.collate_fn_multiple_size)
        valid_gallery_loader = DataLoader(gallery_valid, batch_size=batch_size, collate_fn=du.collate_fn_multiple_size, num_workers=prefetch)
        test_queries_loader = DataLoader(queries_test, batch_size=1, collate_fn=du.collate_fn_multiple_size)
        test_gallery_loader = DataLoader(gallery_test, batch_size=batch_size, collate_fn=du.collate_fn_multiple_size, num_workers=prefetch)
        node_size=2
        return train_loader, valid_queries_loader, valid_gallery_loader, test_queries_loader, test_gallery_loader, node_size
    raise NameError(dataset + ' not implemented!')


def print_statistics(data_train, data_valid, gallery_valid, data_test, gallery_test):
    print('* Train with {} groups of graphs'.format(len(data_train)))
    print('* Validation with {} queries and {} graphs in the gallery'.format(len(data_valid), len(gallery_valid)))
    print('* Test with {} queries and {} graphs in the gallery'.format(len(data_test), len(gallery_test)))


def load_iam(data_path, triplet=False):
    from .Iam import Iam_train, Iam, create_graph_iam
    # Split path
    split = os.path.normpath(data_path).split(os.sep)
    split[-2] = split[-2] + '-pickled'
    pickle_dir = os.path.join(*split)
    if split[0]=='':
        pickle_dir = os.sep + pickle_dir

    if not os.path.isdir(pickle_dir):
        # Data to pickle
        dataset_to_pickle(data_path, pickle_dir, create_graph_iam, '.gxl')

    # Get data for train, validation and test
    data_train = Iam_train(pickle_dir, os.path.join(data_path, 'train.cxl'), triplet)

    if os.path.isfile(os.path.join(data_path, 'validation.cxl')):
        data_valid = Iam(pickle_dir, os.path.join(data_path, 'validation.cxl'))
    else:
        data_valid = Iam(pickle_dir, os.path.join(data_path, 'valid.cxl'))

    data_test = Iam(pickle_dir, os.path.join(data_path, 'test.cxl'))
    gallery = Iam(pickle_dir, os.path.join(data_path, 'train.cxl'))
    return data_train, data_valid, data_test, gallery


def load_histograph_gw(data_path, triplet=False, set_partition='cv1'):
    from .HistoGraph import HistoGraph_train, HistoGraph, create_graph_histograph
    # Split path
    split = os.path.normpath(data_path).split(os.sep)
    split[-2] = split[-2] + '-pickled'
    pickle_dir = os.path.join(*split)
    if split[0]=='':
        pickle_dir = os.sep + pickle_dir

    if not os.path.isdir(pickle_dir):
        # Data to pickle
        dataset_to_pickle(data_path, pickle_dir, create_graph_histograph, '.gxl')

    gt_path = os.path.join(data_path, os.pardir, '00_GroundTruth', set_partition)
    data_train = HistoGraph_train(pickle_dir, os.path.join(gt_path,'train.txt'), triplet)

    gallery_valid = HistoGraph(pickle_dir, os.path.join(gt_path, 'valid.txt'), dataset='gw')
    gallery_test = HistoGraph(pickle_dir, os.path.join(gt_path, 'test.txt'), dataset='gw')

    queries = HistoGraph(pickle_dir, os.path.join(gt_path, 'train.txt'), os.path.join(gt_path, 'keywords.txt'), dataset='gw')
    # Get labels to create a unique identifier
    unique_labels = np.unique(np.concatenate((queries.getlabels(), gallery_valid.getlabels(), gallery_test.getlabels())))
    ulabels_dict = {l:i for i, l in enumerate(unique_labels)}
    gallery_valid.setlabelsdict(ulabels_dict)
    gallery_test.setlabelsdict(ulabels_dict)
    queries.setlabelsdict(ulabels_dict)
    return data_train, queries, gallery_valid, gallery_test


def load_histograph_ak(data_path, triplet=False):
    from .HistoGraph import HistoGraph_train, HistoGraph, create_graph_histograph
    # Split path
    split = os.path.normpath(data_path).split(os.sep)
    split[-2] = split[-2] + '-pickled'
    pickle_dir = os.path.join(*split)
    if split[0]=='':
        pickle_dir = os.sep + pickle_dir

    if not os.path.isdir(pickle_dir):
        # Data to pickle
        dataset_to_pickle(data_path, pickle_dir, create_graph_histograph, '.gxl')

    gt_path = os.path.join(data_path, os.pardir, '00_GroundTruth' )
    data_train = HistoGraph_train(os.path.join(pickle_dir, '01_Train_I'), os.path.join(gt_path, '01_Train_I', 'words.txt'), triplet)

    gallery_valid = HistoGraph(os.path.join(pickle_dir, '01_Train_I'), os.path.join(gt_path, '01_Train_I', 'words.txt'), dataset='ak')
    gallery_test = HistoGraph(os.path.join(pickle_dir, '02_Test'), os.path.join(gt_path, '02_Test', 'words.txt'))

    queries_valid = HistoGraph(os.path.join(pickle_dir, '01_Train_I'), os.path.join(gt_path, '01_Train_I', 'words.txt'), os.path.join(gt_path, '02_Test', 'queries.txt'), dataset='ak')
    queries_test = HistoGraph(os.path.join(pickle_dir, '02_Test'), os.path.join(gt_path, '02_Test', 'queries.txt'))

    # Get labels to create a unique identifier
    unique_labels = np.unique(np.concatenate((queries_valid.getlabels(), queries_test.getlabels(), gallery_valid.getlabels(), gallery_test.getlabels())))
    ulabels_dict = {l:i for i, l in enumerate(unique_labels)}
    gallery_valid.setlabelsdict(ulabels_dict)
    gallery_test.setlabelsdict(ulabels_dict)
    queries_valid.setlabelsdict(ulabels_dict)
    queries_test.setlabelsdict(ulabels_dict)
    return data_train, queries_valid, gallery_valid, queries_test, gallery_test


def dataset_to_pickle(root_path, out_path, graph_reader, graph_ext):
    os.makedirs(out_path)
    dir_list = glob.glob(os.path.join(root_path, '*/'))
    for d in dir_list:
        dataset_to_pickle(d, os.path.join(out_path,list(filter(None,d.split(os.sep)))[-1]), graph_reader, graph_ext)
    file_list = glob.glob(os.path.join(root_path, '*'+graph_ext))
    id_list = np.array([re.search(os.path.join(r'^'+root_path, '(.*)'+graph_ext+'$'), s).group(1) for s in file_list])
    for f in tqdm(id_list):
        node_labels, am = graph_reader(os.path.join(root_path, f + graph_ext))
        graph_dict = {'node_labels': node_labels, 'am': am}
        pickle_file = os.path.join(out_path, f+'.p')
        pickle.dump(graph_dict, open(pickle_file, "wb"))

