#-*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import dgl

"""
    Useful data tools.
"""

__author__ = 'Pau Riba'
__email__ = 'priba@cvc.uab.cat'


def collate_fn_multiple_size(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    gdata = list(map(lambda g: g.gdata['std'], graphs))
    batched_graph = dgl.batch(graphs)
    batched_graph.gdata = {'std': torch.stack(gdata)}
    return batched_graph, labels


def collate_fn_multiple_size_siamese(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).

    # Check if we are in triplet setting
    triplet = False
    if samples[0][3].size()[0] == 0:
        triplet = True

    if triplet:
        graphs1, graphs2, graphs3, labels = map(list, zip(*samples))

        gdata = list(map(lambda g: g.gdata['std'], graphs1))
        batched_graph1 = dgl.batch(graphs1)
        batched_graph1.gdata = {'std': torch.stack(gdata)}

        gdata = list(map(lambda g: g.gdata['std'], graphs2))
        batched_graph2 = dgl.batch(graphs2)
        batched_graph2.gdata = {'std': torch.stack(gdata)}

        gdata = list(map(lambda g: g.gdata['std'], graphs3))
        batched_graph3 = dgl.batch(graphs3)
        batched_graph3.gdata = {'std': torch.stack(gdata)}

        return batched_graph1, batched_graph2, batched_graph3, None

    graphs1, graphs2, _, labels = map(list, zip(*samples))

    gdata = list(map(lambda g: g.gdata['std'], graphs1))
    batched_graph1 = dgl.batch(graphs1)
    batched_graph1.gdata = {'std': torch.stack(gdata)}

    gdata = list(map(lambda g: g.gdata['std'], graphs2))
    batched_graph2 = dgl.batch(graphs2)
    batched_graph2.gdata = {'std': torch.stack(gdata)}

    return batched_graph1, batched_graph2, None, torch.tensor(labels)

