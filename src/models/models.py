import torch
import torch.nn as nn
from .layers import GConv, EdgeCompute

class GNN(nn.Module):
    def __init__(self, in_feat, out_feat, nlayers=3, hid=256, dropout=True, J=2):
        super(GNN, self).__init__()
        self.nlayers = nlayers
        self.hid = hid

        # Embedding Function
        self.embedding = nn.Linear(in_feat, self.hid)

        self.J = J
        for i in range(1, nlayers):
            # Compute learned connections up to order self.J
            module_wc = EdgeCompute(i*self.hid, self.hid, J=self.J)
            self.add_module('wc{}'.format(i), module_wc)
            
            # Graph convolution
            module_gc = GConv(i*self.hid, self.hid, J=self.J)
            self.add_module('gc{}'.format(i), module_gc)

        # Last operation to map to output size
        self.wc_last = EdgeCompute(self.nlayers*self.hid, self.hid, J=self.J)
        self.fc_last = nn.Linear(self.nlayers*self.hid, out_feat, bias=True)
        
        self.dropout = nn.Dropout()
        self.nl = nn.ReLU()

    def forward(self, g):
        # Unpack graph
        x, Win, g_size = g

        # Initialize self connections
        Wid = [self._wid(x.size(0))]

        # Embedd node positions to higher space
        x = self.dropout(self.nl(self.embedding(x)))

        for i in range(1, self.nlayers):
            # List of adjacency information up to order self.J
            W = self._modules['wc{}'.format(i)](x, Win)

            # Graph Convolution
            x_new = self.nl(self._modules['gc{}'.format(i)](x, Wid + W))
            
            # Dropout
            x_new = self.dropout(x_new)
            
            # Concat information at different steps
            x = torch.cat([x, x_new], 1)

        # Last layer
        W = self.wc_last(x, Win)
        x = self.fc_last(x)

        # W[0] contains the learned values of Win
        return (x, W[0], g_size)

    def _wid(self, s):
        # Identity matrix (self connections) of size s
        i = torch.LongTensor(range(s))
        i = torch.stack((i,i))
        Wid = torch.sparse.FloatTensor(i, torch.ones(s), torch.Size([s,s]))
        if next(self.parameters()).is_cuda:
            Wid = Wid.cuda()
        return Wid


if __name__=="__main__":
    net = GNN(3, 20)
    print(net)

