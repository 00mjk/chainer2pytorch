# chainer2pytorch

`chainer2pytorch` implements conversions from Chainer modules to PyTorch modules,
setting parameters of each modules such that one can port over models on a module basis.

Installation:

```bash
pip install git+git://github.com/vzhong/chainer2pytorch.git
```


Usage:

```python
from tc import nn
from chainer import links as L

c = L.Linear(10, 50)
t = nn.Linear.from_chainer(c)

c = L.NStepLSTM(1, 10, 20, 0)
t = nn.from_chainer(c)
```

Note that when do you a forward call, PyTorch's `LSTM` only gives the
output of the last layer, whereas chainer gives the output of all layers.

Test:

```bash
nosetests tests
```

Pull requests are welcome!