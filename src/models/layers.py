"""Torch Module for Gated Graph Convolution layer"""
import torch
from torch import nn
from torch.nn import init

from dgl import function as fn

class EdgeConv(nn.Module):
    def __init__(self, in_feat, out_feat, residual=False, activation=torch.sigmoid):
        super(EdgeConv, self).__init__()

        self.residual = residual

        h_feat = 64
        self.mlp =  nn.Sequential(   nn.Linear(in_feat,h_feat),
                                     nn.ReLU(True),
                                     nn.Linear(h_feat, out_feat))

        self.activation = activation

    def message(self, edges):
        e_out = self.mlp((edges.dst['x']-edges.src['x']).abs())

        if self.residual:
            e_out = edges.data['x'] + e_out

        if self.activation is not None:
            e_out = self.activation(e_out)

        return {'e': e_out}

    def forward(self, g, h, he=None):
        with g.local_scope():

            g.ndata['x'] = h
            if he is not None:
                g.edata['x'] = he

            g.apply_edges(self.message)

            return g.edata['e']


class GatedGraphConv(nn.Module):
    r"""Gated Graph Convolution layer from paper `Gated Graph Sequence
    Neural Networks <https://arxiv.org/pdf/1511.05493.pdf>`__.

    .. math::
        h_{i}^{0} & = [ x_i \| \mathbf{0} ]

        a_{i}^{t} & = \sum_{j\in\mathcal{N}(i)} W_{e_{ij}} h_{j}^{t}

        h_{i}^{t+1} & = \mathrm{GRU}(a_{i}^{t}, h_{i}^{t})

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    n_steps : int
        Number of recurrent steps.
    edge_func : callable activation function/layer
        Maps each edge feature to a vector of shape
        ``(in_feats * out_feats)`` as weight to compute
        messages.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 n_steps,
                 edge_func,
                 edge_feats,
                 bias=True,
                 dropout = 0.3,
                 aggregator_type='sum'):

        super(GatedGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._n_steps = n_steps
        self.edge_nn = edge_func
        if aggregator_type == 'sum':
            self.reducer = fn.sum
        elif aggregator_type == 'mean':
            self.reducer = fn.mean
        elif aggregator_type == 'max':
            self.reducer = fn.max
        else:
            raise KeyError('Aggregator type {} not recognized: '.format(aggregator_type))
        self.aggre_type = aggregator_type
        self.gru = nn.GRUCell(in_feats, out_feats, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = init.calculate_gain('relu')
        self.gru.reset_parameters()

    def forward(self, graph, feat, efeat):
        """Compute Gated Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`N`
            is the number of nodes of the graph and :math:`D_{in}` is the
            input feature size.
        etypes : torch.LongTensor
            The edge type tensor of shape :math:`(E,)` where :math:`E` is
            the number of edges of the graph.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is the output feature size.
        """
        graph = graph.local_var()

        for step in range(self._n_steps):
            # (n, d_in, 1)
            graph.ndata['h'] = feat.unsqueeze(-1)
            # (n, d_in, d_out)
            graph.edata['w'] = self.edge_nn(efeat).view(-1, self._in_feats, self._out_feats)
            graph.update_all(fn.u_mul_e('h', 'w', 'm'), self.reducer('m', 'neigh'))

            rst = graph.ndata.pop('neigh').sum(dim=1) # (N, D)

            feat = self.gru(rst, feat)
            if step < self._n_steps-1:
                feat = self.dropout(feat)
        return feat
