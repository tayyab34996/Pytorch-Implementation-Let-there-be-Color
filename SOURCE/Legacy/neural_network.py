# -*- coding: utf-8 -*-
"""
PyTorch implementation of the neural network building blocks.
This mirrors the classes and behavior used by the original TensorFlow
version (activations, shapes and fusion logic) but implemented with
PyTorch `torch` operations.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import config


class Layer(object):

    def __init__(self, shape, stddev, value):
        # Original shape convention: [kH, kW, inC, outC]
        self.shape = shape
        self.stddev = stddev
        self.value = value


class Convolution_Layer(Layer):

    def __init__(self, shape, stddev, value):
        super(Convolution_Layer, self).__init__(shape, stddev, value)
        kH, kW, inC, outC = shape
        weight = torch.empty(outC, inC, kH, kW)
        nn.init.normal_(weight, mean=0.0, std=stddev)
        bias = torch.full((outC,), value)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def feed_forward(self, input_data, stride):
        # input_data expected in NCHW format
        s = stride[1]
        padding = (self.shape[0] // 2, self.shape[1] // 2)
        conv = F.conv2d(input_data, self.weight, bias=self.bias, stride=s, padding=padding)
        output_data = torch.tanh(conv)
        return output_data


class FullyConnected_Layer(Layer):

    def __init__(self, shape, stddev, value):
        super(FullyConnected_Layer, self).__init__(shape, stddev, value)
        inF, outF = shape
        weight = torch.empty(inF, outF)
        nn.init.normal_(weight, mean=0.0, std=stddev)
        bias = torch.full((outF,), value)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def feed_forward(self, input_data, stride=None):
        # input_data expected shape [B, inF]
        fullyconnected = input_data.matmul(self.weight)
        output_data = F.relu(fullyconnected + self.bias)
        return output_data


class Fusion_Layer(Convolution_Layer):

    def __init__(self, shape, stddev, value):
        super(Fusion_Layer, self).__init__(shape, stddev, value)

    def feed_forward(self, mid_features, global_features, stride):
        # mid_features: NCHW -> convert to NHWC-like for the fusion logic
        # global_features: [B, F]
        B, C, H, W = mid_features.shape
        # move to [B, H*W, C]
        mid = mid_features.permute(0, 2, 3, 1).reshape(B, H * W, C)
        # expand global to [B, H*W, F]
        global_exp = global_features.unsqueeze(1).expand(-1, H * W, -1)
        # concat -> [B, H*W, C+F]
        fusion = torch.cat([mid, global_exp], dim=2)
        # reshape to [B, H, W, C+F]
        fusion = fusion.reshape(B, H, W, -1)
        # move to NCHW for convolution
        fusion = fusion.permute(0, 3, 1, 2)
        return super(Fusion_Layer, self).feed_forward(fusion, stride)


class Output_Layer(Layer):

    def __init__(self, shape, stddev, value):
        super(Output_Layer, self).__init__(shape, stddev, value)
        kH, kW, inC, outC = shape
        weight = torch.empty(outC, inC, kH, kW)
        nn.init.normal_(weight, mean=0.0, std=stddev)
        bias = torch.full((outC,), value)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def feed_forward(self, input_data, stride):
        s = stride[1]
        padding = (self.shape[0] // 2, self.shape[1] // 2)
        conv = F.conv2d(input_data, self.weight, bias=self.bias, stride=s, padding=padding)
        output_data = torch.sigmoid(conv)
        return output_data
