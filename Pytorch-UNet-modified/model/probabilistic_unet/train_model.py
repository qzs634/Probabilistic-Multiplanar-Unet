import torch
import numpy as np
import logging
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from load_LIDC_data import LIDC_IDRI
from mri_dataset import MRI_Dataset
from probabilistic_unet import ProbabilisticUnet
from utils import l2_regularisation
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

img_dir = r"C:\Users\Niklas Magnussen\Desktop\TheBachelor\data_folder\train\images"
mask_dir = r"C:\Users\Niklas Magnussen\Desktop\TheBachelor\data_folder\train\labels"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#dataset = LIDC_IDRI(dataset_location = 'data/')
dataset = MRI_Dataset(img_dir, mask_dir, 4)
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.1 * dataset_size))
train, val = random_split(dataset, [dataset_size - split, split])
train_loader = DataLoader(train, batch_size=5, shuffle=True, num_workers=5, pin_memory=True, drop_last=True)

"""
for batch in train_loader:
    img = batch['image'] # [b, c, h, w]
    mask = batch['mask']

    for b in range(img.shape[0]):
        if torch.max(img[b,:,:,:]) == 0:
            print("Something is very wrong!")
            
"""

"""
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(dataset, batch_size=5, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler)
logging.info(f"Number of training/test patches: {len(train_indices)} / {len(test_indices)}")
"""

writer = SummaryWriter()

net = ProbabilisticUnet(input_channels=1, num_classes=4, num_filters=[64,128,256,512], latent_dim=6, no_convs_fcomb=4, beta=10.0)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)
epochs = 10
global_step = 0
for epoch in range(epochs):
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        for step, batch in enumerate(train_loader):
            patch = batch["image"]
            patch = patch.to(device)
            mask = batch["mask"]
            mask = mask.to(device)
            #mask = torch.unsqueeze(mask,1)
            net.forward(patch, mask, training=True)
            elbo = net.elbo(mask)
            reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
            loss = -elbo + 1e-5 * reg_loss
            writer.add_scalar("elbo_loss", loss.item(), global_step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if (global_step % 100) == 0:
                with torch.no_grad():
                    colors = [torch.Tensor([0., 0., 0.]), torch.Tensor([1., 0., 0.]), torch.Tensor([0., 1., 0.]),
                              torch.Tensor([0., 0., 1.])]

                    pred_mask = net.sample()

                    writer.add_images('images', patch, global_step)

                    batch, _, h, w = pred_mask.shape
                    pred_mask_img = torch.zeros((batch, h, w, 3))
                    true_mask_img = torch.zeros((batch, h, w, 3))
                    pred_idx = torch.argmax(pred_mask, dim=1)
                    for b in range(batch):
                        for i in range(h):
                            for j in range(w):
                                pred_mask_img[b, i, j] = colors[pred_idx[b, i, j]]
                                true_mask_img[b, i, j] = colors[int(mask.squeeze(1)[b, i, j])]
                    writer.add_images('masks/true', true_mask_img, global_step, dataformats='NHWC')
                    writer.add_images('masks/pred', pred_mask_img, global_step, dataformats='NHWC')

            global_step += 1
            pbar.update(1)