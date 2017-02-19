import unittest
from tc import nn
import torch
from torch.autograd import Variable
from chainer import links as L
import numpy as np


class TestLSTM(unittest.TestCase):
    batch_size = 3
    d_in = 2
    d_hid = 3

    def test_forward(self):
        n_layers = 1
        x = np.random.uniform(0, 1, [self.batch_size, self.d_in])

        c = L.NStepLSTM(n_layers, self.d_in, self.d_hid, 0)
        t = nn.LSTM.from_chainer(c)

        tx = Variable(torch.Tensor(x.tolist()))
        h0 = c0 = Variable(tx.data.new(n_layers, self.batch_size, self.d_hid).fill_(0))

        ch, cc, _ = c(h0.data.numpy(), c0.data.numpy(), list(tx.data.numpy()[:, None, :]))
        _, [th, tc] = t(tx.unsqueeze(0), [h0, c0])

        self.assertTrue(np.allclose(ch.data, th.data.numpy())), 'ch:\n{}\nth:\n{}'.format(ch.data, th.data.numpy())
        self.assertTrue(np.allclose(cc.data, tc.data.numpy())), 'cc:\n{}\ntc:\n{}'.format(cc.data, tc.data.numpy())


class TestLinear(TestLSTM):

    def test_forward(self):
        x = np.random.uniform(0, 1, [self.batch_size, self.d_in]).astype(np.float32)
        c = L.Linear(self.d_in, self.d_hid, bias=np.random.uniform(0, 1, [self.d_hid]))
        t = nn.Linear.from_chainer(c)

        tx = Variable(torch.Tensor(x.tolist()))

        co = c(x)
        to = t(tx)
        self.assertTrue(np.allclose(co.data, to.data.numpy()), 'co:\n{}\nto:\n{}'.format(co.data, to.data.numpy()))


class TestEmbedding(TestLSTM):

    seq_len = 5
    vocab_size = 10
    d_emb = 2

    def test_forward(self):
        x = np.random.randint(0, self.vocab_size, [self.batch_size, self.seq_len]).astype(np.int32)
        c = L.EmbedID(self.vocab_size, self.d_emb)
        t = nn.Embedding.from_chainer(c)

        tx = Variable(torch.LongTensor(x.tolist()))

        co = c(x)
        to = t(tx)
        self.assertTrue(np.allclose(co.data, to.data.numpy()), 'co:\n{}\nto:\n{}'.format(co.data, to.data.numpy()))


if __name__ == '__main__':
    unittest.main()
