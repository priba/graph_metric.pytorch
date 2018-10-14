import torch
import torch.nn as nn
from .layers import GConv

class GNN(nn.Module):
    def __init__(self, in_feat, out_feat, nlayers=3, dropout=True):
        super(GNN, self).__init__()
        self.nlayers = nlayers
        self.hid = 256
        self.embedding = nn.Linear(in_feat, self.hid)
        for i in range(1, nlayers):
            module_gc = GConv(i*self.hid, self.hid)
            self.add_module('gc{}'.format(i), module_gc)
        self.gc_last = GConv(self.nlayers*self.hid, out_feat, bias_bool=False, bn_bool=False)
        self.dropout = nn.Dropout()
        self.nl = nn.LeakyReLU()

    def forward(self, g):
        x, W, g_size = g
        Wid = self._wid(x.size(0))

        x = self.nl(self.embedding(x))

        for i in range(1, self.nlayers):
            x_new = self.nl(self._modules['gc{}'.format(i)](x, [Wid, W]))
            x = torch.cat([x, x_new], 1)

        x = self.dropout(x)
        x = self.gc_last(x, [Wid, W])

        return (x, W, g_size)

    def _wid(self, s):
        i = torch.LongTensor(range(s))
        i = torch.stack((i,i))
        Wid = torch.sparse.FloatTensor(i, torch.ones(s), torch.Size([s,s]))
        if next(self.parameters()).is_cuda:
            Wid = Wid.cuda()
        return Wid


if __name__=="__main__":
    net = GNN(3, 20)
    print(net)
