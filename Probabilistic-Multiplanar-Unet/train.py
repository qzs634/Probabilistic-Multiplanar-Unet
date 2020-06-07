import argparse
import logging
import os
import sys
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from model import UNet
from dice_loss import dice_coeff
from trainer import UNetTrainer, ProbUNetTrainer

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from utils.mri_dataset import MRI_Dataset, mri_collate
from torch.utils.data import DataLoader, random_split

# Image locations
dir_img  = r"C:\Users\Niklas Magnussen\Desktop\TheBachelor\data_folder\train\images" #"data/imgs/"
dir_mask = r"C:\Users\Niklas Magnussen\Desktop\TheBachelor\data_folder\train\labels" #"data/masks/"
dir_checkpoint = 'checkpoints/'

def train_net(trainer,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              lrf=0.1,
              lrp=2,
              om=0.9,
              val_percent=0.1,
              save_cp=False):

    # Training dataset and validation dataset
    dataset = MRI_Dataset(dir_img, dir_mask, trainer.net.n_classes)
    n_val = int(len(dataset) * val_percent) # Size of validationset
    n_train = len(dataset) - n_val # Size of trainingset
    train, val = random_split(dataset, [n_train, n_val]) # Create training and validationset with given size

    # gradient accumulator steps
    acc_steps = 4 if batch_size > 4 else 1

    # Generate batches for training and loading
    train_loader = DataLoader(train, batch_size=batch_size // acc_steps, shuffle=True, num_workers=6, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size // acc_steps, shuffle=False, num_workers=6, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LRF_{lrf}_LRP_{lrp}_EP_{epochs}_LR_{lr}_BS_{batch_size}')
    global_step = 0
    global_step_size = 1

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    optimizer = optim.SGD(trainer.net.parameters(), lr=lr, momentum=om)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if trainer.net.n_classes > 1 else 'max', factor=lrf, patience=lrp)

    for epoch in range(epochs):

        trainer.net.train()

        with tqdm(total=n_train + n_val, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for phase in ["train", "validation"]:
                if phase == "train":
                    optimizer.zero_grad() # Reset gradient
                    out_loss = 0
                    for i, batch in enumerate(train_loader):
                        imgs = batch['image']
                        true_masks = batch['mask']

                        # Move data to GPU memory
                        imgs = imgs.to(device=trainer.device, dtype=torch.float32)
                        true_masks = true_masks.to(device=device, dtype=trainer.mask_type)

                        masks_pred = trainer.predict(imgs, true_masks)

                        # Clear GPU memory after prediction
                        if device == 'cuda':
                            del imgs
                            del true_masks
                            torch.cuda.empty_cache()

                        # Calculate loss
                        loss = trainer.loss(imgs, true_masks, masks_pred)
                        loss = loss / acc_steps

                        # Calculate gradient according to loss
                        loss.backward()

                        # At accumulated target batch size, take optimizer step
                        # E.g. target batch size is 8, mini batches are size 2, so every 4th iteration take optim step
                        if (i + 1) % acc_steps == 0:
                            out_loss = loss.item()
                            writer.add_scalar('Loss/train', out_loss, global_step)
                            pbar.set_postfix(**{'loss (batch)': out_loss})
                            out_loss = 0

                            nn.utils.clip_grad_value_(trainer.net.parameters(), 0.1)
                            optimizer.step()
                            optimizer.zero_grad()

                            # Clear GPU memory
                            if device == 'cuda':
                                torch.cuda.empty_cache()

                        global_step += global_step_size
                        pbar.update(imgs.shape[0])

                elif phase == "validation":
                    pbar.set_description(f'Epoch {epoch + 1}/{epochs} (validation round)')

                    optimizer.zero_grad() # Reset gradient

                    trainer.net.eval()

                    val_count = len(val_loader)  # the number of batches in validation set

                    # Array of dice scores for each class, except for background
                    dices = 0
                    dice_sums = np.array([0.0] * (trainer.net.n_classes - 1)) # [0] * 3 -> [0, 0, 0]
                    loss_sum = 0

                    for batch in val_loader:
                        imgs, true_masks = batch['image'], batch['mask']

                        # Move data to GPU
                        imgs = imgs.to(device=device, dtype=torch.float32)
                        true_masks = true_masks.to(device=device, dtype=trainer.mask_type)

                        # No gradient calculations are necessary when doing validation
                        with torch.no_grad():
                            masks_pred = trainer.predict(imgs, true_masks) # [bg, c1, c2, c3]

                        # Add the new dice scores to each class element wise
                        dice = trainer.eval(imgs, true_masks, masks_pred)

                        if trainer.net.n_classes > 1:
                            dice_sums += dice
                        else:
                            dices += dice

                        # Calculate validation loss
                        loss_sum += trainer.loss(imgs, true_masks, masks_pred).item()

                        # Write a single image during validation
                        if (global_step % val_count) == 0:
                            writer.add_images('images', imgs, global_step)

                            writer.add_images('masks/true', trainer.mask_to_image(true_masks), global_step)
                            writer.add_images('masks/pred', trainer.mask_to_image(masks_pred, prediction=True), global_step)


                        global_step += global_step_size
                        pbar.update()

                    # write metrics
                    avg_loss = loss_sum / val_count
                    writer.add_scalar('Loss/validation', avg_loss, global_step)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    for c in range(trainer.net.n_classes - 1):
                        writer.add_scalar(f'dice/class_{c + 1}', dice_sums[c] / val_count, global_step)

                    #Adjust learning rate based on metric
                    if trainer.net.n_classes == 1:
                        val_score = (dices / val_count)[0]
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('metrics/dice', val_score, global_step)
                    else:
                        val_score = avg_loss

                    scheduler.step(val_score)

                # End of epoch
                # Save checkpoint
                torch.save(trainer.net.state_dict(),
                           dir_checkpoint + trainer.name + f'_checkpoint{epoch}.pt')
                logging.info(f'Saved model {trainer.name}_checkpoint{epoch}.pt')

                gc.collect()

    # End of training
    torch.save(trainer.net.state_dict(),
               dir_checkpoint + trainer.name + '_model.pt')
    logging.info(f'Saved model {trainer.name}_model.pt')
    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-r', '--schedule-factor', metavar='LRF', type=float, nargs='?', default=0.1,
                        help='Learning rate scheduler factor', dest='lrf')
    parser.add_argument('-p', '--schedule-patience', metavar='LRP', type=int, nargs='?', default=5,
                        help='Learning rate scheduler patience', dest='lrp')
    parser.add_argument('-o', '--optimizer-momentum', metavar='OM', type=float, nargs='?', default=0.9,
                        help='Optimizer momentum', dest='om')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-m', '--model', dest='net', type=str, default="unet",
                        help='what model to use: unet or probunet')
    parser.add_argument('-d', '--dir', dest='dir', type=str, default=None,
                        help='image and label superdirs.')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device} with properties= {torch.cuda.get_device_properties(device)}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N

    if args.net == "unet":
        trainer = UNetTrainer(device, n_channels=1, n_classes=1, load_model=args.load)
    elif args.net == "probunet":
        trainer = ProbUNetTrainer(device, n_channels=1, n_classes=3, load_model=args.load, latent_dim=6, beta=10)
    else:
        print("Error! {} is not a valid model".format(args.net))

    if args.dir is not None:
        dir_img = os.path.join(args.dir, "images")
        dir_mask = os.path.join(args.dir, "labels")

    dataset = MRI_Dataset(dir_img, dir_mask, 3)

    # Faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(trainer,
                      epochs=args.epochs,
                      batch_size=args.batchsize,
                      lr=args.lr,
                      lrf= args.lrf,
                      lrp= args.lrp,
                      om=args.om,
                      device=device,
                      val_percent=args.val / 100)
        del trainer
        torch.cuda.empty_cache()
        gc.collect()

    except KeyboardInterrupt:
        torch.save(trainer.net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

