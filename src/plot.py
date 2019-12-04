#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

"""
    Graph metric learning
"""

# Python modules
import torch
import networkx as nx
import glob
import numpy as np
import time
import os
import sys
import copy
import matplotlib.pyplot as plt

# Own modules
from options import Options
from utils import load_checkpoint
from models import models, distance
from data.HistoGraph import HistoGraph
__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"


def plot(g1, g2, ind1, ind2, name1, name2, distance):

    pos1 = dict(g1.nodes(data='h'))
    pos1 = {k: v.numpy() for k, v in pos1.items()}

    pos2 = dict(g2.nodes(data='h'))
    pos2 = {k: v.numpy() for k, v in pos2.items()}

    plt.subplot(221)
    nx.draw(g1, pos=pos1)
    plt.title(f"{name1}")
    plt.gca().invert_yaxis()

    plt.subplot(222)
    nx.draw(g2, pos=pos2)
    plt.title(f"{name2}")
    plt.gca().invert_yaxis()

    plt.subplot(212)

    h = nx.disjoint_union(g1,g2)

    id_g1 = range(len(g1))
    id_g2 = range(len(g1), len(g1)+len(g2))

    pos = dict(h.nodes(data='h'))
    pos = {k: v.numpy() for k, v in pos.items()}

    nx.draw_networkx_nodes(h, pos=pos, nodelist=id_g1, node_size=50,node_color='b')
    nx.draw_networkx_nodes(h, pos=pos, nodelist=id_g2, node_size=50,node_color='c')
    nx.draw_networkx_edges(h, pos=pos)

    # Insertions
    insert_list = [i for i, v in enumerate(ind1) if v==len(g2)]
    # Delitions
    delitions_list = [i+len(g1) for i, v in enumerate(ind2) if v==len(g1)]

    nx.draw_networkx_nodes(h, pos=pos, nodelist=insert_list, node_size=50,node_color='b',edgecolors='lime', linewidths=2.0)
    nx.draw_networkx_nodes(h, pos=pos, nodelist=delitions_list, node_size=50,node_color='c', edgecolors='orange', linewidths=2.0)

    h.remove_edges_from(h.edges())
    h = h.to_directed()

    for i, v in enumerate(ind1):
        if v!=len(g2):
            h.add_edge(i, v+len(g1))
    for i, v in enumerate(ind2):
        if v!=len(g1):
            h.add_edge(i+len(g1), v)

    nx.draw_networkx_edges(h, pos=pos, edge_color='r')
    plt.title(f"{name1}-{name2} D: {distance}")
    plt.gca().invert_yaxis()
    plt.show()


def main(query, query_name,  target, target_name):

    print('Prepare data')
    split = os.path.normpath(args.data_path).split(os.sep)
    split[-2] = split[-2] + '-pickled'
    pickle_dir = os.path.join(*split)
    if split[0]=='':
        pickle_dir = os.sep + pickle_dir
    gt_path = os.path.join(args.data_path, os.pardir, '00_GroundTruth', args.set_partition)
    data = HistoGraph(pickle_dir, os.path.join(gt_path, 'test.txt'))
    data_query = copy.deepcopy(data)
    data_query.graphs = [query]
    data_query.labels = [query_name]
    g1, l1 = data_query[0]


    data_target = data
    data_target.graphs = [target]
    data_target.labels = [target_name]
    g2, l2 = data_target[0]
    del data_target, data_query, data

    print('Create model')
    net = models.GNN(2, args.hidden, args.out_size, dropout=args.dropout)
    distNet = distance.SoftHd(args.out_size)

    print('Check CUDA')
    g1_orig = copy.deepcopy(g1)
    g2_orig = copy.deepcopy(g2)
    if args.cuda:
        print('\t* CUDA')
        net, distNet = net.cuda(), distNet.cuda()
        g1.ndata['h'] = g1.ndata['h'].cuda()
        g2.ndata['h'] = g2.ndata['h'].cuda()


    if args.load is not None:
        print('Loading model')
        checkpoint = load_checkpoint(args.load)
        net.load_state_dict(checkpoint['state_dict'])
        distNet.load_state_dict(checkpoint['state_dict_dist'])

    print('***PLOT***')
    g1_out = net(g1)
    g2_out = net(g2)
    dist, indB, indA = distNet.soft_hausdorff(g1_out, g2_out)
    plot(g1_orig.to_networkx(node_attrs=['h']).to_undirected(), g2_orig.to_networkx(node_attrs=['h']).to_undirected(), indB.tolist(), indA.tolist(), query_name, target_name, dist.item())

if __name__ == '__main__':
    # Parse options
    args = Options().parse()
    print('Parameters:\t' + str(args))

    # Check cuda & Set random seed
    args.cuda = args.ngpu > 0 and torch.cuda.is_available()

    if args.seed > 1:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    # Check Test and Load
    if args.load is None:
        raise Exception('Cannot plot without loading a model.')

    main(query='276-19-06.p', query_name='day', target='306-30-04.p', target_name='day')
    main(query='276-19-06.p', query_name='day', target='306-32-08.p', target_name='the')

