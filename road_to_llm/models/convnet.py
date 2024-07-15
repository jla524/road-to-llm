"""
https://medium.com/@BrendanArtley/mnist-keras-simple-cnn-99-6-731b624aee7f
"""
from tinygrad import nn, Tensor


class ConvNet:
    def __init__(self):
        self.layers = [
            nn.Conv2d(1, 32, 5), Tensor.relu,
            nn.Conv2d(32, 32, 5), Tensor.relu,
            nn.BatchNorm2d(32), Tensor.max_pool2d,
            nn.Conv2d(32, 64, 3), Tensor.relu,
            nn.Conv2d(64, 64, 3), Tensor.relu,
            nn.BatchNorm2d(64), Tensor.max_pool2d,
            lambda x: x.flatten(1), nn.Linear(576, 10),
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)
