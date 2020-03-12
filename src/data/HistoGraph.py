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


class HistoGraph_train(data.Dataset):
    def __init__(self, root_path, file_list, triplet=False):
        self.root = root_path
        self.file_list = file_list
        self.triplet = triplet

        self.graphs, self.labels = getFileList(self.file_list)

        # To pickle
        self.graphs = [g+'.p' for g in self.graphs]
        self.labels = np.array(self.labels)
        self.unique_labels = np.unique(self.labels)
        if self.triplet:
            # Triplet (anchor, positive, negative)
            self.groups = [ (i, j) for i in range(len(self.labels)) for j in np.where(self.labels[i] == self.labels)[0] if i != j ]
            self.labels_len = np.array(list(map(len, self.labels)))
            self.labels_counts = np.array([(l==self.labels).sum() for l in self.labels])
        else:
            # Siamese all pairs
            self.groups = list(itertools.permutations(range(len(self.labels)), 2))

    def __getitem__(self, index):
        ind = self.groups[index]

        # Graph 1
        g1 = self._loadgraph(ind[0])
        target1 = self.labels[ind[0]]

        # Graph 2
        g2 = self._loadgraph(ind[1])
        target2 = self.labels[ind[1]]

        if self.triplet:
            # Random negative choice where it would be of similar size
            possible_ind = np.where(self.labels!=target1)[0]
            labels_counts = self.labels_counts[possible_ind]
            labels_len = np.abs(self.labels_len[possible_ind] - self.labels_len[ind[0]]) + 1.0
            labels_probs = 1/(labels_counts*labels_len)
            labels_probs = labels_probs/labels_probs.sum()
            neg_ind = np.random.choice(possible_ind, 1, p=labels_probs)

            # Graph 3
            g3 = self._loadgraph(neg_ind[0])
            target_neg = self.labels[neg_ind[0]]

            return g1, g2, g3, torch.Tensor([])

        target = torch.FloatTensor([0.0]) if target1 == target2 else torch.FloatTensor([1.0])
        return g1, g2, torch.Tensor([]), target

    def __len__(self):
        return len(self.groups)

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


class HistoGraph(data.Dataset):
    def __init__(self, root_path, file_list, keywords_file=None, dataset='gw'):
        self.root = root_path
        self.file_list = file_list

        self.graphs, self.labels = getFileList(self.file_list)

        # To pickle
        self.graphs = [g+'.p' for g in self.graphs]

        self.dataset = dataset

        if keywords_file is not None:
            with open(keywords_file, 'r') as f:
                queries = f.read().splitlines()
            queries = [ q.split(' ')[-1] for q in queries ]
            idx_del = [i for i, label in enumerate(self.labels) if label not in queries]

            for index in sorted(idx_del, reverse=True):
                del self.labels[index]
                del self.graphs[index]

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

    def getlabels(self):
        return np.unique(self.labels)

    def setlabelsdict(self, lab_dict):
        self.labels_dict = lab_dict


def getFileList(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()

    classes = []
    elements = []
    for line in lines:
        f, c = line.split(' ')[:2]
        classes += [c]
        elements += [f]
    return elements, classes


def create_graph_histograph(file, representation='adj'):

    tree_gxl = ET.parse(file)
    root_gxl = tree_gxl.getroot()
    graph_properties = []
    node_label = []
    node_id = []

    for x_std in root_gxl.iter('attr'):
        if x_std.get('name') == 'x_std' or x_std.get('name') == 'y_std':
            graph_properties.append(float(x_std.find('float').text))

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

    return graph_properties, node_label, am

