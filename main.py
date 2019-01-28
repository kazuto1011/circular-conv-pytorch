#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   28 January 2019

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from libs.models.vgg import CircularConv2d


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":

    input = torch.ones(1, 1, 16, 16).requires_grad_()
    kwargs = {"kernel_size": 3, "stride": 1, "bias": False}

    conv1 = nn.Sequential(
        nn.Conv2d(1, 1, padding=1, **kwargs),
        nn.Conv2d(1, 1, padding=1, **kwargs),
        nn.Conv2d(1, 1, padding=1, **kwargs),
        nn.Conv2d(1, 1, padding=1, **kwargs),
        nn.Conv2d(1, 1, padding=1, **kwargs),
    )
    init_weights(conv1)

    conv2 = nn.Sequential(
        CircularConv2d(1, 1, **kwargs),
        CircularConv2d(1, 1, **kwargs),
        CircularConv2d(1, 1, **kwargs),
        CircularConv2d(1, 1, **kwargs),
        CircularConv2d(1, 1, **kwargs),
    )
    init_weights(conv2)

    # Normal convolution
    conv1(input).sum().backward()
    plt.figure()
    plt.imshow(input.grad.cpu().numpy()[0][0])

    input.grad.zero_()

    # Horizontal circular convolution
    conv2(input).sum().backward()
    plt.figure()
    plt.imshow(input.grad.cpu().numpy()[0][0])

    plt.show()
