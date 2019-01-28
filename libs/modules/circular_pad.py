#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-27


import collections
from itertools import repeat

import torch.nn as nn

from ..functions.circular_pad import CircularPad2d as F_CircularPad2d


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_quadruple = _ntuple(4)


class CircularPad2d(nn.Module):
    def __init__(self, padding=(1, 1, 1, 1)):
        super(CircularPad2d, self).__init__()
        self.padding = _quadruple(padding)

    def forward(self, input):
        return F_CircularPad2d(padding=self.padding)(input)

    def __repr__(self):
        return self.__class__.__name__ + str(self.padding)
