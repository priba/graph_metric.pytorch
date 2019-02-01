import math

import torch
import torch.nn as nn
from torch.autograd import Function


class GConv(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias_bool=True, bn_bool=True, J=2):
        super(GConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.J = J # Number of W
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
                output.append(torch.sparse.mm(w, x))
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

class EdgeCompute(nn.Module):
    """
    Simple Edge computation layer, similar to https://arxiv.org/pdf/1711.04043.pdf
    """

    def __init__(self, in_features, hid=64, J=2):
        super(EdgeCompute, self).__init__()
        self.in_features = in_features
        self.hid = hid
        self.J = J

        # Define a network per each order in the adjacency matrix
        for i in range(1, self.J):
            module_mlp = nn.Sequential(
                    nn.Linear(self.in_features, self.hid),
                    nn.ReLU(),
                    nn.Linear(self.hid, 1),
                    nn.Sigmoid()
                    )
            self.add_module('mlp{}'.format(i), module_mlp)

    def forward(self, x, W):
        Wnew = []
        if W._nnz() == 0:
            for i in range(1, self.J):
                Wnew.append(W)
        else:
            for i in range(1, self.J):
                Wi = W.pow(i)
                indices = Wi._indices()
                data = Wi._values()
                x_diff = x[indices[0]] - x[indices[1]]
                data = self._modules['mlp{}'.format(i)](x_diff.abs()).squeeze()
                Wnew.append(ValuesToSparse.apply(indices, data, W.shape))
        return Wnew

    def __repr__(self):
        return self.__class__.__name__ + '(in_features=' \
               + str(self.in_features) + ')'


class Block(nn.Module):
    def __init__(self, in_features, out_features, J=2):
        super(Block, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.J = J
        self.nl = nn.ReLU()

        # Last operation to map to output size
        self.wc1 = EdgeCompute(self.in_features, 32, J=self.J)
        self.wc2 = EdgeCompute(self.in_features, 32, J=self.J)
        
        self.conv1 = GConv(self.in_features, self.out_features, J=self.J+1)
        
        self.dropout = nn.Dropout(0.3)

        self.conv2 = GConv(self.out_features, self.out_features, J=self.J+1)

        if self.in_features != self.out_features:
            self.conv_res = nn.Linear(self.in_features, self.out_features, bias=False)


    def forward(self, x, Wid, Win):
        W1 = self.wc1(x, Win)
        W2 = self.wc2(x, Win)

        residual = x

        x = self.conv1(x, Wid + W1 + W2)
        x = self.nl(x)

        x = self.dropout(x)

        x = self.conv2(x, Wid + W1 + W2)

        if self.in_features != self.out_features:
            residual = self.conv_res(residual)

        x = x + residual
        x = self.nl(x)

        return x


# Copy Gradients when getting the values of a sparse tensor
class GetValues(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input._values().clone()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        
        grad_input = torch.sparse.FloatTensor(input._indices(), grad_output, input.shape)
        if input.is_cuda:
            grad_input = grad_input.cuda()
        return grad_input


# Create a new torch Float Tensor and allow the gradients to backpropagate
class ValuesToSparse(Function):
    @staticmethod
    def forward(ctx, indices, data, shape):
        ctx.save_for_backward(indices, data)
        output = torch.sparse.FloatTensor(indices, data, shape).requires_grad_(True)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        indices, input = ctx.saved_tensors
        grad_input = None
        grad_output = grad_output.coalesce()
        grad_input = grad_output._values().clone()
        return None, grad_input, None

