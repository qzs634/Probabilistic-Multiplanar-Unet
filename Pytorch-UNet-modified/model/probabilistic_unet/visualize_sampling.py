import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import logging
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from load_LIDC_data import LIDC_IDRI
from mri_dataset import MRI_Dataset
from probabilistic_unet import ProbabilisticUnet

def visualize_sample(net, slice):
    """
    load a model
    load a slice
    use the model to sample n predictions from the slice
    plt.show
    """

    predictions = [net.predictions]
