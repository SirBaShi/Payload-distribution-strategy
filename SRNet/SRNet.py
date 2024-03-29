import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class MSRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(MSRBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        return x + x1 + x2 + x3 + x4


class SRNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, num_phases, num_blocks):
        super(SRNetGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_phases = num_phases
        self.num_blocks = num_blocks
        self.encoder = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.decoder = nn.Conv2d(out_channels, in_channels, 3, padding=1)
        self.sparse_net = nn.ModuleList([MSRBlock(out_channels, out_channels) for _ in range(num_blocks)])
        self.image_net = nn.ModuleList([MSRBlock(in_channels, in_channels) for _ in range(num_blocks)])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.encoder(x)
        s = torch.zeros_like(h) 
        y = torch.zeros_like(x) 
        for i in range(self.num_phases):
            for j in range(self.num_blocks):
                s = self.sparse_netj))
            for k in range(self.num_blocks):
                y = self.image_netk))
        return y, s

class SRNetDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(SRNetDiscriminator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=2)
        self.conv3 = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in range(num_layers-2)])
        self.conv4 = nn.Conv2d(out_channels, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x)) 
        x = self.relu(self.conv2(x)) 
        for i in range(self.num_layers-2):
            x = self.relu(self.conv3i) 
        x = self.sigmoid(self.conv4(x))
        return x
