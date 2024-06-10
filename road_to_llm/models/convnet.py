"""
https://medium.com/@BrendanArtley/mnist-keras-simple-cnn-99-6-731b624aee7f
"""
from torch import nn, tensor
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.c3 = nn.Conv2d(32, 64, 3)
        self.bn3 = nn.BatchNorm2d(64)
        self.c4 = nn.Conv2d(64, 64, 3)
        self.bn4 = nn.BatchNorm2d(64)
        self.l1 = nn.Linear(1024, 128)
        self.l2 = nn.Linear(128, 10)

    def __call__(self, x: tensor) -> tensor:
        x = self.bn1(self.c1(x).relu())
        x = self.bn2(self.c2(x).relu())
        x = F.max_pool2d(x, 2, stride=2)
        x = self.bn3(self.c3(x).relu())
        x = self.bn4(self.c4(x).relu())
        x = F.max_pool2d(x, 2, stride=2)
        x = x.flatten(1)
        x = self.l1(x).relu()
        x = self.l2(x)
        return x
