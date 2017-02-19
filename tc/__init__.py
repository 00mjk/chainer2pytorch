import torch
from torch import autograd
import numpy as np


class Variable(autograd.Variable):

    @classmethod
    def from_chainer(cls, c):
        T = torch.LongTensor if 'int' in str(c.dtype) else torch.Tensor
        t = cls(T(c.data.tolist()), volatile=bool(c.volatile))
        return t
