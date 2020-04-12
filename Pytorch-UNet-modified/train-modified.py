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
from torch.utils.data import DataLoader, random_split, ConcatDataset

dir_img = 'data/imgs/'
dir_mask = 'data/masks/'
dir_checkpoint = 'checkpoints/'

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              lrf=0.1,
              lrp=2,
              om=0.9,
              k=4,
              one_fold=True,
              save_cp=True,
              img_scale=0.5):

    dataset = BasicDataset(dir_img, dir_mask, net.n_classes, img_scale)
    n = int(len(dataset) / k)
    splits = random_split(dataset, [n for _ in range(k)])

    logging.info(f'''Starting training:
                Epochs:          {epochs}
                Batch size:      {batch_size}
                Learning rate:   {lr}
                Training size:   {n * (k - 1)}
                Validation size: {n}
                Checkpoints:     {save_cp}
                Device:          {device.type}
                Images scaling:  {img_scale}
            ''')

    folds = 1 if one_fold else len(splits)
    for i in range(folds):
        # Reset learning parameters (weights and biases)
        net = UNet(net.n_channels, net.n_classes)
        net.to(device=device)

        # Cycle trough training set and validation set in cross validation
        train_set = ConcatDataset(np.take(splits, [j for j in range(i, i + k - 1)], mode='wrap', axis=0))
        val_set = splits[(i + k - 1) % k]

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

        writer = SummaryWriter(comment=f'_EP_{epochs}_LR_{lr}_LRF_{lrf}_LRP_{lrp}_OM_{om}_BS_{batch_size}')
        global_step = 0

        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=om)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', factor=lrf, patience=lrp) #'min' if net.n_classes > 1 else 'max'
        criterion = nn.BCELoss() if net.n_classes == 1 else nn.CrossEntropyLoss()

        for epoch in range(epochs):
            print("Epoch {} out of {}".format(epoch + 1, epochs))

            for phase in ["train", "validation"]:
                if phase == "train":
                    net.train()
                    print("Training round")
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
                        #masks_pred = masks_pred.squeeze(1)
                        loss = criterion(masks_pred, true_masks)

                        writer.add_scalar('Loss/train', loss.item(), global_step)

                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_value_(net.parameters(), 0.1)
                        optimizer.step()

                        global_step += 1

                elif phase == "validation":
                    print("Validation round")
                    optimizer.zero_grad()
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        if not value.grad == None:
                            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                    net.eval()
                    mask_type = torch.float32
                    n_val = len(val_loader)  # the number of images in batch
                    loss_sum = 0
                    dice_sum = 0
                    accuracy_sum = 0
                    sensitivity_sum = 0
                    specificity_sum = 0
                    for batch in val_loader:
                        imgs, true_masks = batch['image'], batch['mask']
                        imgs = imgs.to(device=device, dtype=torch.float32)
                        true_masks = true_masks.to(device=device, dtype=mask_type)

                        with torch.no_grad():
                            masks_pred = net(imgs)

                        # Calculate validation loss
                        loss_sum += criterion(masks_pred, true_masks).item()

                        if net.n_classes == 1:
                            # Calculate dice score for each
                            dice_sum += dice_coeff((masks_pred > 0.5).float(), true_masks).item()

                        # Calculate accuracy, sensitivity and specificity scalars
                        confusion_vector = torch.round(masks_pred) / true_masks
                        true_positives = torch.sum(confusion_vector == 1).item() # 1/1
                        false_positives = torch.sum(confusion_vector == float('-inf')).item() + torch.sum(confusion_vector == float('inf')).item() # 1/0
                        true_negatives = torch.sum(torch.isnan(confusion_vector)).item() # 0/0
                        false_negatives = torch.sum(confusion_vector == 0).item() # 0/1

                        # print("true positives: {}\ntrue negatives: {}\nfalse positives: {}\nfalse negatives: {}\n".format(true_positives, true_negatives, false_positives, false_negatives))
                        if (true_positives + true_negatives + false_negatives + false_positives) > 0:
                            accuracy_sum += (true_positives + true_negatives) / (
                            true_positives + true_negatives + false_positives + false_negatives)
                        if (true_positives + false_negatives) > 0:
                            sensitivity_sum += true_positives / (true_positives + false_negatives)
                        if (true_negatives + false_positives) > 0:
                            specificity_sum += true_negatives / (true_negatives + false_positives)

                        if (global_step % n_val) == 0:
                            writer.add_images('images', imgs, global_step)
                            if net.n_classes == 1:
                                writer.add_images('masks/true', true_masks * 255, global_step)
                                writer.add_images('masks/pred', (masks_pred > 0.5) * 255, global_step)

                        global_step += 1

                    # Write average of metrics
                    loss = loss_sum / n_val
                    writer.add_scalar('Loss/validation', loss_sum / n_val, global_step)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                    writer.add_scalar('metrics/accuracy', accuracy_sum / n_val, global_step)
                    writer.add_scalar('metrics/sensitivity', sensitivity_sum / n_val, global_step)
                    writer.add_scalar('metrics/specificity', specificity_sum / n_val, global_step)

                    #Adjust learning rate based on average dice score for epoch
                    val_score = dice_sum / n_val if net.n_classes == 1 else loss
                    logging.info('Validation Dice Coeff: {}'.format(val_score))
                    writer.add_scalar('metrics/dice', val_score, global_step)
                    scheduler.step(val_score)

        # save model
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'split_{i}_EP_{epochs}_LR_{lr}_LRF_{lrf}_LRP_{lrp}_OM_{om}_BS_{batch_size}.pth')
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
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-r', '--lr-factor', metavar='LRF', type=float, nargs='?', default=0.1,
                        help='Learning rate schedule factor', dest='lrf')
    parser.add_argument('-t', '--lr-patience', metavar='LRP', type=int, nargs='?', default=2,
                        help='Learning rate schedule patience', dest='lrp')
    parser.add_argument('-m', '--optimizer-momentum', metavar='OM', type=float, nargs='?', default=0.9,
                        help='Optimizer momentum', dest='om')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-k', '--k-split', dest='k', type=int, default=2,
                        help='Amount of splits of the dataset')
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
    net = UNet(n_channels=1, n_classes=3)
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
                  lrf=args.lrf,
                  lrp=args.lrp,
                  om=args.om,
                  device=device,
                  img_scale=args.scale,
                  k=args.k)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
