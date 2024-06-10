"""
https://github.com/geohot/ai-notebooks/blob/master/mnist_from_scratch.ipynb
"""
from torch import nn, tensor


class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_features: int = 128):
        super().__init__()
        self.l1 = nn.Linear(in_features, hidden_features)
        self.l2 = nn.Linear(hidden_features, out_features)
        self.in_features = in_features

    def __call__(self, x: tensor) -> tensor:
        x = x.reshape(-1, self.in_features)
        x = self.l1(x).relu()
        x = self.l2(x)
        return x
