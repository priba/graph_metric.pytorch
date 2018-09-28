import torch.nn as nn
from layers import GConv

class GNN(nn.Module):
    def __init__(self, in_feat, nclass, dropout=True):
        super(GNN, self).__init__()

        self.gc1 = GConv(in_feat, 64)
        self.gc2 = GConv(64, nclass)
        self.dropout = nn.Dropout()
        self.nl = nn.LeakyReLU()

    def forward(self, x, W):
        x = self.nl(self.gc1(x, W))
        x = self.dropout(x)
        x = self.gc2(x, W)
        return x


if __name__=="__main__":
    net = GNN(3, 20)
    print(net)
