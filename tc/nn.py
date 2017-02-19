import torch
from torch import nn
from chainer import functions as F


class LSTM(nn.LSTM):

    @classmethod
    def from_chainer(cls, c):
        cweights = c._children[0]
        d_in, d_hid = cweights.w0.shape
        t = cls(d_in, d_hid)
        c_ih_w = F.concat([getattr(cweights, 'w{}'.format(i)) for i in range(4)], 0).data
        c_ih_b = F.concat([getattr(cweights, 'b{}'.format(i)) for i in range(4)], 0).data
        t.weight_ih_l0.data = torch.from_numpy(c_ih_w)
        t.bias_ih_l0.data = torch.from_numpy(c_ih_b)

        c_hh_w = F.concat([getattr(cweights, 'w{}'.format(i)) for i in range(4, 8)], 0).data
        c_hh_b = F.concat([getattr(cweights, 'b{}'.format(i)) for i in range(4, 8)], 0).data
        t.weight_hh_l0.data = torch.from_numpy(c_hh_w)
        t.bias_hh_l0.data = torch.from_numpy(c_hh_b)
        return t


class Linear(nn.Linear):

    @classmethod
    def from_chainer(cls, c):
        d_hid, d_in = c.W.shape
        t = cls(d_in, d_hid)
        t.weight.data = torch.from_numpy(c.W.data)
        t.bias.data = torch.from_numpy(c.b.data)
        return t

    def forward(self, input):
        shape = list(input.size())
        d_hid, d_in = self.weight.size()
        shape[-1] = d_hid
        return super().forward(input.view(-1, d_in)).view(*shape)


class Embedding(nn.Embedding):

    @classmethod
    def from_chainer(cls, c):
        vocab_size, d_emb = c.W.shape
        t = cls(vocab_size, d_emb)
        t.weight.data = torch.from_numpy(c.W.data)
        return t
