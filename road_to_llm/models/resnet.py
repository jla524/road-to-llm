"""
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
"""
from tinygrad import nn


class Bottleneck:
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, dilation=dilation, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def __call__(self, x):
        residual = x
        x = self.bn1(self.conv1(x)).relu()
        x = self.bn2(self.conv2(x)).relu()
        x = self.bn3(self.conv3(x))
        if self.downsample is not None:
            residual = residual.sequential(self.downsample)
        x = x + residual
        return x.relu()


class ResNet:
    def __init__(self, block, layers, num_classes=1000, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.layer1 = self.__make_layer(block, 64, layers[0])
        self.layer2 = self.__make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self.__make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self.__make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def __make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = [
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            ]
        layers = [block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return layers

    def __call__(self, x):
        x = self.bn1(self.conv1(x)).relu()
        x = x.max_pool2d(kernel_size=(3, 3), stride=2, padding=1)
        x = x.sequential(self.layer1)
        x = x.sequential(self.layer2)
        x = x.sequential(self.layer3)
        x = x.sequential(self.layer4)
        x = x.avg_pool2d(kernel_size=(1, 1))
        x = x.flatten(1)
        return self.fc(x)


def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model
