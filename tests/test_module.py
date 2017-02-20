from tc import softmax
from chainer import functions as F
import numpy as np
import unittest
import torch
from torch.autograd import Variable


class TestSoftmax(unittest.TestCase):

    def test_call(self):
        x = np.random.uniform(0, 1, [1, 2, 3])
        c = F.softmax(x).data
        t = softmax(Variable(torch.Tensor(x.tolist()))).data.numpy()
        self.assertTrue(np.allclose(c, t), 'c:\n{}\nt:\n{}'.format(c, t))


