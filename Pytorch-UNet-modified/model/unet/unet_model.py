""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, num_filters=[64,128,256,512], bilinear=False, apply_last_layer=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.apply_last_layer = apply_last_layer
        self.num_filters = num_filters
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        self.inc = DoubleConv(n_channels, self.num_filters[0])
        self.outc = OutConv(self.num_filters[0], n_classes)
        for i in range(len(self.num_filters) - 1):
            print(f"down({self.num_filters[i]}, {self.num_filters[i + 1]})\nup({self.num_filters[i + 1] * 2}, {self.num_filters[i + 1]})")
            self.down_blocks.append(Down(self.num_filters[i], self.num_filters[i+1]))
            self.up_blocks.append(Up(self.num_filters[i + 1] * 2, self.num_filters[i + 1]))
        self.up_blocks = self.up_blocks[::-1]

    def forward(self, x):

        print(f"inc = {self.inc}")

        xs = []
        xs.append(self.inc(x))
        for i in range(len(self.down_blocks)):
            xs.append(self.down_blocks[i](xs[i]))

        for i in range(len(self.up_blocks)):
            print(f"arg1: {xs[-1].shape}, arg2: {xs[-(2 + i *2)].shape}")
            xs.append(self.up_blocks[i](xs[-1], xs[-(2 + i * 2)]))
        xs.append(self.outc(xs[-1]))


        if self.n_classes == 1:
            xs[-1] = torch.sigmoid(xs[-1])

        if self.apply_last_layer:
            return xs[-1]
        else:
            return xs[-2]
