import torch
from torch import autograd
from torch.nn import functional as F


class Variable(autograd.Variable):

    @classmethod
    def from_chainer(cls, c):
        T = torch.LongTensor if 'int' in str(c.dtype) else torch.Tensor
        t = cls(T(c.data.tolist()), volatile=bool(c.volatile))
        return t


def softmax(a):
    shape = list(a.size())
    last_dim = len(shape) - 1
    flat = a.transpose(1, last_dim).contiguous().view(-1, shape[1])
    out = F.softmax(flat)
    shape[1], shape[last_dim] = shape[last_dim], shape[1]
    return out.view(*shape).transpose(1, last_dim)
