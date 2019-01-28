#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      https://github.com/kazuto1011
# Created:  2017-04-11

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.circular_pad import CircularPad2d


def _init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=1e-2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class ConvBlock(nn.Sequential):
    def __init__(self, n_in, n_out):
        super(ConvBlock, self).__init__()
        kwargs = {"kernel_size": 3, "stride": 1, "padding": 1, "bias": False}
        self.add_module("conv", nn.Conv2d(n_in, n_out, **kwargs))
        self.add_module("norm", nn.BatchNorm2d(n_out))
        self.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        return super(ConvBlock, self).forward(x)


class CircularConv2d(nn.Sequential):
    def __init__(self, n_in, n_out, **kwargs):
        super(CircularConv2d, self).__init__()
        self.add_module("cpad", CircularPad2d(padding=(1, 1, 1, 1)))
        self.add_module("conv", nn.Conv2d(n_in, n_out, padding=0, **kwargs))

    def forward(self, x):
        return super(CircularConv2d, self).forward(x)


class CircularConvBlock(nn.Sequential):
    def __init__(self, n_in, n_out):
        super(CircularConvBlock, self).__init__()
        kwargs = {"kernel_size": 3, "stride": 1, "bias": False}
        self.add_module("conv", CircularConv2d(n_in, n_out, **kwargs))
        self.add_module("norm", nn.BatchNorm2d(n_out))
        self.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        return super(CircularConvBlock, self).forward(x)


class VGG(nn.Module):
    def __init__(self, n_in):
        super(VGG, self).__init__()
        ch = [n_in, 64, 128, 256, 512, 512, 512]
        self.features = nn.Sequential(
            OrderedDict(
                [
                    # Layer1
                    ("block1", ConvBlock(ch[0], ch[1])),
                    ("pool1", nn.MaxPool2d(kernel_size=2)),
                    # Layer2
                    ("block2", ConvBlock(ch[1], ch[2])),
                    ("pool2", nn.MaxPool2d(kernel_size=2)),
                    # Layer3
                    ("block3_1", ConvBlock(ch[2], ch[3])),
                    ("block3_2", ConvBlock(ch[3], ch[3])),
                    ("pool3", nn.MaxPool2d(kernel_size=2)),
                    # Layer4
                    ("block4_1", ConvBlock(ch[3], ch[4])),
                    ("block4_2", ConvBlock(ch[4], ch[4])),
                    ("pool4", nn.MaxPool2d(kernel_size=2)),
                    # Layer5
                    ("block5_1", ConvBlock(ch[4], ch[5])),
                    ("block5_2", ConvBlock(ch[5], ch[5])),
                    ("pool5", nn.MaxPool2d(kernel_size=2)),
                ]
            )
        )
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(ch[5] * 12, ch[6], bias=False)),
                    ("norm1", nn.BatchNorm1d(ch[6])),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("drop1", nn.Dropout(p=0.5)),
                    ("fc2", nn.Linear(ch[6], 6, bias=False)),
                ]
            )
        )
        _init_weights(self)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class PanoramicVGG(nn.Module):
    def __init__(self, n_in):
        super(PanoramicVGG, self).__init__()
        ch = [n_in, 64, 128, 256, 512, 512, 512]
        self.features = nn.Sequential(
            OrderedDict(
                [
                    # Layer1
                    ("block1", CircularConvBlock(ch[0], ch[1])),
                    ("pool1", nn.MaxPool2d(kernel_size=2)),
                    # Layer2
                    ("block2", CircularConvBlock(ch[1], ch[2])),
                    ("pool2", nn.MaxPool2d(kernel_size=2)),
                    # Layer3
                    ("block3_1", CircularConvBlock(ch[2], ch[3])),
                    ("block3_2", CircularConvBlock(ch[3], ch[3])),
                    ("pool3", nn.MaxPool2d(kernel_size=2)),
                    # Layer4
                    ("block4_1", CircularConvBlock(ch[3], ch[4])),
                    ("block4_2", CircularConvBlock(ch[4], ch[4])),
                    ("pool4", nn.MaxPool2d(kernel_size=2)),
                    # Layer5
                    ("block5_1", CircularConvBlock(ch[4], ch[5])),
                    ("block5_2", CircularConvBlock(ch[5], ch[5])),
                    # Row-Wise Max Pooling
                    ("rwmp", nn.MaxPool2d((1, 24))),
                ]
            )
        )
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(ch[5] * 2, ch[6], bias=False)),
                    ("norm1", nn.BatchNorm1d(ch[6])),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("drop1", nn.Dropout(p=0.5)),
                    ("fc2", nn.Linear(ch[6], 6, bias=False)),
                ]
            )
        )
        _init_weights(self)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

