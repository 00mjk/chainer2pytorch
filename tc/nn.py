import torch
import chainer
from torch import nn
from chainer import functions as F

from chainer.functions.activation.lstm import _extract_gates


def replace_weight(t, c):
    t.data.copy_(torch.from_numpy(c.data))

def split_clstm(var):
    return [chainer.Variable(m[0]) for m in _extract_gates(var.data[None, :])]

class LSTMCell(nn.LSTMCell):

    @classmethod
    def from_chainer(cls, c):
        c_wa_ih, c_wi_ih, c_wf_ih, c_wo_ih = split_clstm(c.upward.W)
        c_wa_hh, c_wi_hh, c_wf_hh, c_wo_hh = split_clstm(c.lateral.W)
        c_ba, c_bi, c_bf, c_bo = split_clstm(c.upward.b)
        d_hid, d_in = c_wa_ih.shape
        t = cls(d_in, d_hid)
        c_w_ih = F.concat([c_wi_ih, c_wf_ih, c_wa_ih, c_wo_ih], axis=0)
        c_w_hh = F.concat([c_wi_hh, c_wf_hh, c_wa_hh, c_wo_hh], axis=0)
        c_b = F.concat([c_bi, c_bf, c_ba, c_bo], axis=0)

        replace_weight(t.weight_ih, c_w_ih)
        replace_weight(t.weight_hh, c_w_hh)
        replace_weight(t.bias_ih, c_b)
        t.bias_hh.data.zero_()

        return t

class LSTM(nn.LSTM):

    @classmethod
    def from_chainer(cls, c):
        cweights = c._children[0]
        d_hid, d_in = cweights.w0.shape
        t = cls(d_in, d_hid)
        c_ih_w = F.concat([getattr(cweights, 'w{}'.format(i)) for i in range(4)], 0)
        c_ih_b = F.concat([getattr(cweights, 'b{}'.format(i)) for i in range(4)], 0)
        replace_weight(t.weight_ih_l0, c_ih_w)
        replace_weight(t.bias_ih_l0, c_ih_b)

        c_hh_w = F.concat([getattr(cweights, 'w{}'.format(i)) for i in range(4, 8)], 0)
        c_hh_b = F.concat([getattr(cweights, 'b{}'.format(i)) for i in range(4, 8)], 0)
        replace_weight(t.weight_hh_l0, c_hh_w)
        replace_weight(t.bias_hh_l0, c_hh_b)
        return t


class Linear(nn.Linear):

    @classmethod
    def from_chainer(cls, c):
        d_hid, d_in = c.W.shape
        t = cls(d_in, d_hid, bias=c.b is not None)
        replace_weight(t.weight, c.W)
        if c.b is not None:
            replace_weight(t.bias, c.b)
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
        replace_weight(t.weight, c.W)
        return t


class Maxout(nn.Module):

    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = Linear(d_in, d_out * pool_size)

    @classmethod
    def from_chainer(cls, c):
        d_in, d_out, pool_size = c.linear.W.shape[-1], c.out_size, c.pool_size
        t = cls(d_in, d_out, pool_size)
        replace_weight(t.lin.weight, c.linear.W)
        replace_weight(t.lin.bias, c.linear.b)
        return t

    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        last_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(last_dim)
        return m.squeeze(last_dim)
