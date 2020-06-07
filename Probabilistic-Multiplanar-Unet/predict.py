import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset

def predict(net, imgs, masks, train=True, prob=False):

    if prob:
        net.forward(imgs, masks, training=train)
        pred_mask = net.sample(testing=(not train))
