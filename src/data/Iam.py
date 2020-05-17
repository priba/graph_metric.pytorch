# -*- coding: utf-8 -*-
import torch
import torch.utils.data as data
import xml.etree.ElementTree as ET
import numpy as np
from . import data_utils as du
import os
import itertools
import pickle
import dgl

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"


class Iam_train(data.Dataset):
    def __init__(self, root_path, file_list, triplet, num_samples=None):
        self.root = root_path
        self.file_list = file_list
        self.triplet = triplet
        self.graphs, self.labels = getFileList(self.file_list)
        # To pickle
        self.graphs = [os.path.splitext(g)[0]+'.p' for g in self.graphs]

        self.unique_labels = np.unique(self.labels)
        self.labels = [np.where(target == self.unique_labels)[0][0] for target in self.labels]
        self.unique_labels = np.unique(self.labels)
        if self.triplet:
            # Triplet (anchor, positive, negative)
            self.groups = [ (i, j) for i in range(len(self.labels)) for j in np.where(self.labels[i] == self.labels)[0] if i != j ]
        else:
            # Siamese all pairs
            self.groups = list(itertools.permutations(range(len(self.labels)), 2))

        if num_samples is not None:
            np.random.shuffle(self.groups)
            num_labels = len(self.unique_labels)
            # Balance positive samples
            pos_samples = num_samples//2
            pos_samples_class = pos_samples//num_labels
            pos_count = np.zeros(self.unique_labels.shape)

            neg_samples = num_samples//2
            neg_samples_class = 2*neg_samples//num_labels
            neg_count = np.zeros(self.unique_labels.shape)

            group = []

            if self.triplet:
                pos_samples_class *= 2
                for gr in self.groups:
                    if pos_count[self.labels[gr[0]] == self.unique_labels] < pos_samples_class:
                        pos_count[self.labels[gr[0]] == self.unique_labels] += 1
                        possible_ind = np.where(self.labels!=self.labels[gr[0]])[0]
                        neg_ind = np.random.choice(possible_ind, 1)[0]
                        while neg_count[self.labels[neg_ind]] >= neg_samples_class:
                            possible_ind = np.where(self.labels!=self.labels[gr[0]])[0]
                            neg_ind = np.random.choice(possible_ind, 1)[0]
                        neg_count[self.labels[neg_ind]] += 1
                        group.append((gr[0], gr[1], neg_ind))
                self.group = group
            else:
                for gr in self.groups:
                    pair_label = self.labels[gr[0]] == self.labels[gr[1]]
                    if pair_label:
                        if pos_count[self.labels[gr[0]] == self.unique_labels] < pos_samples_class:
                            pos_count[self.labels[gr[0]] == self.unique_labels] += 1
                            group.append(gr)
                    else:
                        if (neg_count[self.labels[gr[0]] == self.unique_labels] < neg_samples_class) and (neg_count[self.labels[gr[1]] == self.unique_labels] < neg_samples_class):
                            neg_count[self.labels[gr[0]] == self.unique_labels] += 1
                            neg_count[self.labels[gr[1]] == self.unique_labels] += 1
                            group.append(gr)
                if len(group)<num_samples:
                    for gr in self.groups:
                        if len(group)==num_samples:
                            break
                        if (neg_count[self.labels[gr[0]] == self.unique_labels] < neg_samples_class) or (neg_count[self.labels[gr[1]] == self.unique_labels] < neg_samples_class):
                            neg_count[self.labels[gr[0]] == self.unique_labels] += 1
                            neg_count[self.labels[gr[1]] == self.unique_labels] += 1
                            group.append(gr)
                self.groups = group

    def __getitem__(self, index):
        ind = self.groups[index]

        # Graph 1
        g1 = self._loadgraph(ind[0])
        target1 = self.labels[ind[0]]

        # Graph 2
        g2 = self._loadgraph(ind[1])
        target2 = self.labels[ind[1]]

        if self.triplet:
            if len(ind)==3:
                neg_ind = ind[2]
            else:
                possible_ind = np.where(self.labels!=target1)[0]
                neg_ind = np.random.choice(possible_ind, 1)[0]

            # Graph 3
            g3 = self._loadgraph(neg_ind)
            target_neg = self.labels[neg_ind]

            return g1, g2, g3, torch.Tensor([])

        target = torch.FloatTensor([1.0]) if target1 == target2 else torch.FloatTensor([0.0])
        return g1, g2, torch.Tensor([]), target

    def _loadgraph(self, i):
        graph_dict = pickle.load( open(os.path.join(self.root, self.graphs[i]), "rb") )

        g = dgl.DGLGraph()

        g.gdata = {}
        g.gdata['std'] = torch.tensor(graph_dict['graph_properties']).float()

        g.add_nodes(graph_dict['node_labels'].shape[0])
        g.ndata['pos'] = torch.tensor(graph_dict['node_labels']).float()
        if g.number_of_nodes() == 0:
            g.add_nodes(1, {'pos': torch.zeros(1,2)})
            g.gdata['std'] = torch.zeros(2)


        g.add_edges(graph_dict['am'][0], graph_dict['am'][1])

        # Add self connections
        g.add_edges(g.nodes(), g.nodes())

        return g

    def __len__(self):
        return len(self.groups)


class Iam(data.Dataset):
    def __init__(self, root_path, file_list):
        self.root = root_path
        self.file_list = file_list
        self.graphs, self.labels = getFileList(self.file_list)

        # To pickle
        self.graphs = [os.path.splitext(g)[0]+'.p' for g in self.graphs]

        self.unique_labels = np.unique(self.labels)
        self.labels = [np.where(target == self.unique_labels)[0][0] for target in self.labels]
        self.dataset = 'iam'

    def __getitem__(self, index):
        # Graph
        g = self._loadgraph(index)
        target = self.labels[index]

        return g, target

    def __len__(self):
        return len(self.labels)

    def _loadgraph(self, i):
        graph_dict = pickle.load( open(os.path.join(self.root, self.graphs[i]), "rb") )

        g = dgl.DGLGraph()

        g.gdata = {}
        g.gdata['std'] = torch.tensor(graph_dict['graph_properties']).float()

        g.add_nodes(graph_dict['node_labels'].shape[0])
        g.ndata['pos'] = torch.tensor(graph_dict['node_labels']).float()
        if g.number_of_nodes() == 0:
            g.add_nodes(1, {'pos': torch.zeros(1,2)})
            g.gdata['std'] = torch.zeros(2)

        g.add_edges(graph_dict['am'][0], graph_dict['am'][1])

        return g


def getFileList(file_path):
    elements = []
    classes = []
    tree = ET.parse(file_path)
    root = tree.getroot()
    for child in root:
        for sec_child in child:
            if sec_child.tag == 'print':
                elements += [sec_child.attrib['file']]
                classes += [sec_child.attrib['class']]
    return elements, classes


def create_graph_iam (file):

    tree_gxl = ET.parse(file)
    root_gxl = tree_gxl.getroot()
    node_label = []
    node_id = []
    for node in root_gxl.iter('node'):
        node_id += [node.get('id')]
        for attr in node.iter('attr'):
            if (attr.get('name') == 'x'):
                if attr.find('float') is not None:
                    x = float(attr.find('float').text)
                elif attr.find('int') is not None:
                    x = float(attr.find('int').text)
                else:
                    x = float(attr.find('Integer').text)
            elif (attr.get('name') == 'y'):
                if attr.find('float') is not None:
                    y = float(attr.find('float').text)
                elif attr.find('int') is not None:
                    y = float(attr.find('int').text)
                else:
                    y = float(attr.find('Integer').text)
        node_label += [[x, y]]

    node_label = np.array(node_label)
    node_id = np.array(node_id)

    row, col = np.array([]), np.array([])
    for edge in root_gxl.iter('edge'):
        s = np.where(np.array(node_id)==edge.get('from'))[0][0]
        t = np.where(np.array(node_id)==edge.get('to'))[0][0]

        row = np.append(row, s)
        col = np.append(col,t)

        row = np.append(row, t)
        col = np.append(col,s)

    data = np.ones(row.shape)

    am = row, col, data
    # No properties are provided
    node_label = node_label - node_label.mean(0)
    graph_properties = node_label.std(0)
    node_label = node_label / node_label.std(0)

    return graph_properties, node_label, am

