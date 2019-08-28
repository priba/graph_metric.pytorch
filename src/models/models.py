import torch
import torch.nn as nn
from .layers import GConv, EdgeCompute, Block

class GNN_Orig(nn.Module):
    def __init__(self, in_feat, out_feat, nlayers=3, hid=256, dropout=True, J=2):
        super(GNN_Orig, self).__init__()
        self.nlayers = nlayers
        self.hid = hid
 
        # Embedding Function
        self.embedding = nn.Linear(in_feat, 64, bias=True)
        self.bn = nn.BatchNorm1d(64)
        
        self.J = J

        self.block1 = Block(64, 64, J=self.J)
        self.block2 = Block(64, 64, J=self.J)
        self.block3 = Block(64, 128, J=self.J)
        self.block4 = Block(128, 128, J=self.J)

        # Last operation to map to output size
        self.wc_last = EdgeCompute(64, 32, J=self.J)
        self.fc_last = nn.Linear(64, out_feat)
        
        self.dropout = nn.Dropout(0.3)
        self.nl = nn.ReLU()

    def forward(self, g):
        # Unpack graph
        x, Win, g_size = g

        # Initialize self connections
        Wid = [self._wid(x.size(0))]
        # Embedd node positions to higher space
        x = self.embedding(x)
        x = self.bn(x)
        x = self.nl(x)

        x = self.block1(x, Wid, Win)
        x = self.block2(x, Wid, Win)
        x = self.block3(x, Wid, Win)
        x = self.block4(x, Wid, Win)

        # Last layer
        W = self.wc_last(x, Win)
        x = self.fc_last(x)
        x = nn.Tanh()(x)
        # x = x / x.pow(2).sum(1, keepdim=True).sqrt()
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


class GNN(nn.Module):
    def __init__(self, in_feat, out_feat, nlayers=3, hid=256, dropout=True, J=5):
        super(GNN, self).__init__()
        self.nlayers = nlayers
        self.hid = hid
 
        # Embedding Function
        self.embedding = nn.Linear(in_feat, self.hid, bias=False)
        
        self.J = J

        # Last operation to map to output size
        self.wc0 = EdgeCompute(self.hid, self.hid, J=self.J)
        self.wc1 = EdgeCompute(self.hid, self.hid, J=self.J)
        self.wc2 = EdgeCompute(self.hid, self.hid, J=self.J)
        #self.gc_start = GConv(self.hid, self.hid, J=self.J, bn_bool=True)

        #for i in range(1, nlayers-1):
            # Compute learned connections up to order self.J
        #    module_wc = EdgeCompute(self.hid, self.hid, J=self.J)
        #    self.add_module('wc{}'.format(i), module_wc)
            
            # Graph convolution
        #    module_gc = GConv(self.hid, self.hid, J=self.J, bn_bool=True)
        #    self.add_module('gc{}'.format(i), module_gc)

        # Last operation to map to output size
        self.gc0 = GConv(self.hid, self.hid, J=3*self.J - 2, bn_bool=False)
        self.fc_last = nn.Linear(self.hid, out_feat)
        
        self.dropout = nn.Dropout(0.3)
        self.nl = nn.ReLU()

    def forward(self, g):
        # Unpack graph
        x, Win, g_size = g

        # Initialize self connections
        Wid = [self._wid(x.size(0))]
        # Embedd node positions to higher space
        x = self.embedding(x)
        
        W0 = self.wc0(x, Win)
        W1 = self.wc1(x, Win*Win)
        W2 = self.wc2(x, Win*Win*Win)
        #x = self.dropout(self.nl(self.gc_start(x, Wid + W)))
        #for i in range(1, self.nlayers-1):
            # List of adjacency information up to order self.J
        #    W = self._modules['wc{}'.format(i)](x, Win)

            # Graph Convolution
        #    x_new = self.nl(self._modules['gc{}'.format(i)](x, Wid + W))
            
            # Dropout
        #    x_new = self.dropout(x_new)
            
            # Concat information at different steps
            # x = torch.cat([x, x_new], 1)
        #    x = x_new

        # Last layer
        #x = self.nl(self.gc_last(x, Wid + W0 + W1 + W2))
        x = self.nl(self.gc0(x, Wid + W0 + W1 + W2))
        x = self.fc_last(x)
        # x = nn.Sigmoid()(x)
        # x = x / x.pow(2).sum(1, keepdim=True).sqrt()
        # W[0] contains the learned values of Win
        return (x, Win, g_size)

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

