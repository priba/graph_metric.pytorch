import math

import torch
import torch.nn as nn


class GConv(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias_bool=True, bn_bool=True, J=2):
        super(GConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.J = 2 # Number of W
        self.bias_bool = bias_bool
        self.fc = nn.Linear(self.J*self.in_features, self.out_features, bias=self.bias_bool)

        self.bn_bool = bn_bool
        if self.bn_bool:
            self.bn = nn.BatchNorm1d(self.out_features)

    
    def forward(self, x, W):
        output = []
        for w in W:
            if w._nnz()==0:
                out = torch.zeros(x.shape)
                if x.is_cuda:
                    out = out.cuda()
                output.append(out)
            else:
                output.append(torch.spmm(w, x))
        output = torch.cat(output, dim=1)
        output = self.fc(output)

        if self.bn_bool:
            output = self.bn(output)
        
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(in_features=' \
               + str(self.in_features) + ', out_features=' \
               + str(self.out_features) + ', bias=' \
               + str(self.bias_bool) + ', batch_norm=' \
               + str(self.bn_bool) + ')'

