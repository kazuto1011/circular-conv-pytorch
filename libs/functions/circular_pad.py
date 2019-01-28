#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-27


from torch.autograd import Function


class CircularPad2d(Function):
    def __init__(self, padding):
        super(CircularPad2d, self).__init__()
        self.pad = padding

    def forward(self, input):
        assert input.dim() == 4, "only 4D supported for padding"
        pad_l, pad_r, pad_t, pad_b = self.pad
        h = input.size(2) + pad_t + pad_b
        w = input.size(3) + pad_l + pad_r
        assert w > 0 and h > 0, "input is too small"

        self.input_size = input.size()

        output = input.new(input.size(0), input.size(1), h, w).zero_()

        # crop output if necessary
        c_output = output

        if pad_t > 0:
            c_output = c_output.narrow(2, pad_t, c_output.size(2) - pad_t)
        if pad_b > 0:
            c_output = c_output.narrow(2, 0, c_output.size(2) - pad_b)

        # circular padding
        c_output[:, :, :, 0:pad_l] = input[:, :, :, -pad_r:]
        c_output[:, :, :, -pad_r:] = input[:, :, :, 0:pad_l]

        if pad_l > 0:
            c_output = c_output.narrow(3, pad_l, c_output.size(3) - pad_l)
        if pad_r > 0:
            c_output = c_output.narrow(3, 0, c_output.size(3) - pad_r)
        c_output.copy_(input)

        return output

    def backward(self, grad_output):
        pad_l, pad_r, pad_t, pad_b = self.pad

        grad_input = grad_output.new(self.input_size).zero_()

        cg_input = grad_input

        # crop grad_output if necessary
        cg_output = grad_output
        if pad_t > 0:
            cg_output = cg_output.narrow(2, pad_t, cg_output.size(2) - pad_t)
        if pad_b > 0:
            cg_output = cg_output.narrow(2, 0, cg_output.size(2) - pad_b)
        if pad_l > 0:
            cg_output = cg_output.narrow(3, pad_l, cg_output.size(3) - pad_l)
        if pad_r > 0:
            cg_output = cg_output.narrow(3, 0, cg_output.size(3) - pad_r)
        cg_input.copy_(cg_output)

        cg_input[:, :, :, 0:pad_l] += grad_output[
            :, :, pad_t : grad_output.size(2) - pad_b, -pad_r:
        ]
        cg_input[:, :, :, -pad_r:] += grad_output[
            :, :, pad_t : grad_output.size(2) - pad_b, 0:pad_l
        ]

        return grad_input
