import torch.nn as nn
from .layers import GConv

class GNN(nn.Module):
    def __init__(self, in_feat, out_feat, dropout=True):
        super(GNN, self).__init__()

        self.gc1 = GConv(in_feat, 64)
        self.gc2 = GConv(64, out_feat)
        self.dropout = nn.Dropout()
        self.nl = nn.LeakyReLU()

    def forward(self, g):
        x, W, g_size = g
        x = self.nl(self.gc1(x, W))
        x = self.dropout(x)
        x = self.gc2(x, W)
        return (x, W, g_size)


if __name__=="__main__":
    net = GNN(3, 20)
    print(net)
