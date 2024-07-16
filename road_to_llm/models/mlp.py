"""
https://github.com/geohot/ai-notebooks/blob/master/mnist_from_scratch.ipynb
"""
from tinygrad import nn, tensor


class MLP:
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128, 10)

    def __call__(self, x: tensor) -> tensor:
        x = x.reshape(-1, 784)
        x = self.l1(x).relu()
        x = self.l2(x)
        return x
