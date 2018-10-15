# -*- coding: utf-8 -*-
import torch
import torch.utils.data as data
import xml.etree.ElementTree as ET
import numpy as np
from . import data_utils as du
import os
import itertools
import pickle

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"


class Letters_train(data.Dataset):
    def __init__(self, root_path, file_list, triplet):
        self.root = root_path
        self.file_list = file_list
        self.triplet = triplet
        self.graphs, self.labels = getFileList(self.file_list)

        # To pickle
        self.graphs = [os.path.splitext(g)[0]+'.p' for g in self.graphs]

        self.unique_labels = np.unique(self.labels)
        self.labels = [np.where(target == self.unique_labels)[0][0] for target in self.labels]
        if self.triplet:
            # Triplet (anchor, positive, negative)
            self.groups = [ (i, j) for i in range(len(self.labels)) for j in np.where(self.labels[i] == self.labels)[0] if i != j ]
        else:
            # Siamese all pairs
            self.groups = list(itertools.permutations(range(len(self.labels)), 2))

    def __getitem__(self, index):
        ind = self.groups[index]
        
        # Graph 1
        node_labels1, am1 = self._loadgraph(ind[0])
        target1 = self.labels[ind[0]]
        
        # Graph 2
        node_labels2, am2 = self._loadgraph(ind[1])
        target2 = self.labels[ind[1]]
        if self.triplet:
            neg_ind = np.random.choice(np.where(self.labels!=target1)[0], 1)
            # Graph 3
            node_labels3, am3 = self._loadgraph(neg_ind[0])
            return (node_labels1, am1), (node_labels2, am2), (node_labels3, am3), torch.Tensor([])
        target = torch.FloatTensor([0.0]) if target1 == target2 else torch.FloatTensor([1.0])
        return (node_labels1, am1), (node_labels2, am2), torch.Tensor([]), target

    def _loadgraph(self, i):
        graph_dict = pickle.load( open(os.path.join(self.root, self.graphs[i]), "rb") )
        return graph_dict['node_labels'], graph_dict['am']

    def __len__(self):
        return len(self.groups)


class Letters(data.Dataset):
    def __init__(self, root_path, file_list):
        self.root = root_path
        self.file_list = file_list
        self.graphs, self.labels = getFileList(self.file_list)

        # To pickle
        self.graphs = [os.path.splitext(g)[0]+'.p' for g in self.graphs]

        self.unique_labels = np.unique(self.labels)
        self.labels = [np.where(target == self.unique_labels)[0][0] for target in self.labels]

    def __getitem__(self, index):
        # Graph
        node_labels, am = self._loadgraph(index)
        target = self.labels[index]
        
        return (node_labels, am), target

    def __len__(self):
        return len(self.labels)

    def _loadgraph(self, i):
        graph_dict = pickle.load( open(os.path.join(self.root, self.graphs[i]), "rb") )
        return graph_dict['node_labels'], graph_dict['am']


def getFileList(file_path):
    elements = []
    classes = []
    tree = ET.parse(file_path)
    root = tree.getroot()
    for child in root:
        for sec_child in child:
            if sec_child.tag == 'print':
                elements += [sec_child.attrib['file']]
                classes += sec_child.attrib['class']
    return elements, classes


def create_graph_letter(file):

    tree_gxl = ET.parse(file)
    root_gxl = tree_gxl.getroot()
    node_label = []
    node_id = []
    for node in root_gxl.iter('node'):
        node_id += [node.get('id')]
        for attr in node.iter('attr'):
            if (attr.get('name') == 'x'):
                x = float(attr.find('float').text)
            elif (attr.get('name') == 'y'):
                y = float(attr.find('float').text)
        node_label += [[x, y]]

    node_label = np.array(node_label)
    node_id = np.array(node_id)

    row = np.array([])
    col = np.array([])
    for edge in root_gxl.iter('edge'):
        s = np.where(np.array(node_id)==edge.get('from'))[0][0]
        t = np.where(np.array(node_id)==edge.get('to'))[0][0]

        row = np.append(row, s)
        col = np.append(col,t)

        row = np.append(row, t)
        col = np.append(col,s)
    
    data = np.ones(row.shape)

    am = row, col, data

    return node_label, am

if __name__ == '__main__':
    from options import Options
    from load_data import load_data

    # Parse options
    args = Options().parse()

    # Dataset
    data_train, data_valid, data_test = load_data(args.dataset, args.data_path, args.representation,
                                                           args.normalization)

    nodes = 0
    edges = 0
    for node_labels, am, target in data_train:
        nodes = nodes + node_labels.size(0)
        if len(list(torch.nonzero(am).size())) > 0:
            edges = edges + torch.nonzero(am).size(0)/2.0

    for node_labels, am, target in data_valid:
        nodes = nodes + node_labels.size(0)
        if len(list(torch.nonzero(am).size()))>0:
            edges = edges + torch.nonzero(am).size(0)/2.0

    for node_labels, am, target in data_test:
        nodes = nodes + node_labels.size(0)
        if len(list(torch.nonzero(am).size())) > 0:
            edges = edges + torch.nonzero(am).size(0)/2.0

    nodes = nodes/(len(data_train)+len(data_valid)+len(data_test)+0.0)
    edges = edges / (len(data_train) + len(data_valid) + len(data_test)+0.0)

    print('Nodes: ' + str(nodes))
    print('Edges: ' + str(edges))
