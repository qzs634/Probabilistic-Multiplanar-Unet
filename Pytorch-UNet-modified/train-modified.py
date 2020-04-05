import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet
from dice_loss import dice_coeff

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

dir_img = 'data/imgs/'
dir_mask = 'data/masks/'
dir_checkpoint = 'checkpoints/'


def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):

    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'_EP_{epochs}_LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1) #'min' if net.n_classes > 1 else 'max'
    criterion = nn.BCELoss() #nn.CrossEntropyLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            """
            Hver epoke:
            - første fase: Kør igennem alle batches i træningsæt
                - kør batch igennem netværk
                - udregn loss mellem pred og true
                - Juster netværk ud fra gradient
            - anden fase: Valider netværk på en enkelt batch i validationsæt
                - Kør validation igennem netværk
                - plot metrics
            """
            for phase in ["train", "validation"]:
                if phase == "train":
                    for batch in train_loader:
                        imgs = batch['image']
                        true_masks = batch['mask']

                        assert imgs.shape[1] == net.n_channels, \
                            f'Network has been defined with {net.n_channels} input channels, ' \
                            f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                            'the images are loaded correctly.'

                        imgs = imgs.to(device=device, dtype=torch.float32)
                        mask_type = torch.float32 if net.n_classes == 1 else torch.long
                        true_masks = true_masks.to(device=device, dtype=mask_type)

                        masks_pred = net(imgs)
                        loss = criterion(masks_pred, true_masks)
                        epoch_loss += loss.item()

                        writer.add_scalar('Loss/train', loss.item(), global_step)
                        pbar.set_postfix(**{'loss (batch)': loss.item()})

                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_value_(net.parameters(), 0.1)
                        optimizer.step()

                        global_step += 1
                        pbar.update(imgs.shape[0])

                elif phase == "validation":
                    """
                    Validation phase
                    """
                    optimizer.zero_grad()
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        if not value.grad == None:
                            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                    with tqdm(total=n_val, desc='Validation round', unit='batch') as val_pbar:
                        net.eval()
                        mask_type = torch.float32
                        n_val = len(val_loader)  # the number of batch
                        tot = 0
                        for batch in val_loader:
                            imgs, true_masks = batch['image'], batch['mask']
                            imgs = imgs.to(device=device, dtype=torch.float32)
                            true_masks = true_masks.to(device=device, dtype=mask_type)

                            with torch.no_grad():
                                mask_pred = net(imgs)

                            # Calculate dice score for each
                            pred = dice_coeff((mask_pred > 0.5).float(), true_masks).item()
                            tot += pred

                            logging.info('Validation Dice Coeff: {}'.format(pred))
                            writer.add_scalar('metrics/dice', pred, global_step)

                            # Calculate validation loss
                            loss = criterion(masks_pred, true_masks)

                            writer.add_scalar('Loss/validation', loss.item(), global_step)
                            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                            # Calculate accuracy, sensitivity and specificity scalars
                            confusion_vector = torch.round(masks_pred) / true_masks
                            true_positives = torch.sum(confusion_vector == 1).item()
                            false_positives = torch.sum(confusion_vector == float('-inf')).item() + torch.sum(confusion_vector == float('inf')).item()
                            true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
                            false_negatives = torch.sum(confusion_vector == 0).item()

                            # print("true positives: {}\ntrue negatives: {}\nfalse positives: {}\nfalse negatives: {}\n".format(true_positives, true_negatives, false_positives, false_negatives))
                            if (true_positives + true_negatives + false_negatives + false_positives) > 0:
                                accuracy = (true_positives + true_negatives) / (
                                true_positives + true_negatives + false_positives + false_negatives)
                                writer.add_scalar('metrics/accuracy', accuracy, global_step)
                            if (true_positives + false_negatives) > 0:
                                sensitivity = true_positives / (true_positives + false_negatives)
                                writer.add_scalar('metrics/sensitivity', sensitivity, global_step)
                            if (true_negatives + false_positives) > 0:
                                specificity = true_negatives / (true_negatives + false_positives)
                                writer.add_scalar('metrics/specificity', specificity, global_step)

                            # Write every 100th image
                            if (global_step % n_val-1) == 0:
                                writer.add_images('images', imgs, global_step)
                                if net.n_classes == 1:
                                    writer.add_images('masks/true', true_masks, global_step)
                                    writer.add_images('masks/pred', masks_pred > 0.5, global_step)

                            global_step += 1
                            val_pbar.update()

                    #Adjust learning rate based on average dice score for epoch
                    val_score = tot / n_val
                    logging.info('Validation Dice Coeff: {}'.format(val_score))
                    scheduler.step(val_score)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    # End of training
    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=1, n_classes=1)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
