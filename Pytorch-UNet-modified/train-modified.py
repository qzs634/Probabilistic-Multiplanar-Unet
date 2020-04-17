import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet
from dice_loss import dice_coeff

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

dir_img = "data/imgs/"
dir_mask = "data/masks/"
dir_checkpoint = 'checkpoints/'


def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              lrf=0.1,
              lrp=2,
              om=0.9,
              val_percent=0.1,
              save_cp=False,
              img_scale=0.5):

    dataset = BasicDataset(dir_img, dir_mask, net.n_classes, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])

    """
    class_count = torch.zeros(net.n_classes, dtype=torch.float32)
    for p in train:
        class_i = torch.unique(p['mask'])
        for i in class_i:
            class_count[int(i)] += 1.

    class_sum = torch.sum(class_count)
    #weights = class_count / class_sum
    sampler = None #torch.utils.data.sampler.WeightedRandomSampler(weights, len(train))
    """

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'_EP_{epochs}_LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
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
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=om)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', factor=lrf, patience=lrp)

    class_weights = torch.FloatTensor([1., 1., 1., 1.]).cuda()
    criterion = nn.BCELoss() if net.n_classes == 1 else nn.CrossEntropyLoss(weight=class_weights)
    mask_type = torch.float32 if net.n_classes == 1 else torch.long


    for epoch in range(epochs):
        net.train()

        with tqdm(total=n_train + n_val, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
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
                        true_masks = true_masks.to(device=device, dtype=mask_type)

                        #print(f"True mask:\n   shape={true_masks.shape}\n    unique={torch.unique(true_masks)}")

                        masks_pred = net(imgs)
                        #print(f"Pred mask:\n    shape={masks_pred.shape}\n    unique={torch.unique(masks_pred)}")

                        #loss = criterion(masks_pred, true_masks.squeeze(1))
                        loss = criterion(masks_pred, true_masks)

                        writer.add_scalar('Loss/train', loss.item(), global_step)
                        pbar.set_postfix(**{'loss (batch)': loss.item()})

                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_value_(net.parameters(), 0.1)
                        optimizer.step()


                        # write trainng image
                        """
                        if net.n_classes > 1 and (global_step % 40) == 0:
                            colors = [torch.Tensor([0., 0., 0.]), torch.Tensor([1., 0., 0.]),
                                      torch.Tensor([0., 1., 0.]), torch.Tensor([0., 0., 1.])]
                            batch, _, h, w = true_masks.shape
                            true_mask_img = torch.zeros((batch, h, w, 3))
                            for b in range(batch):
                                for i in range(h):
                                    for j in range(w):
                                        true_mask_img[b, i, j] = colors[true_masks.squeeze(1)[b, i, j]]
                            writer.add_images('training/image', imgs, global_step)
                            writer.add_images('training/masks', true_mask_img, global_step, dataformats='NHWC')
                        """

                        global_step += global_step_size
                        pbar.update(imgs.shape[0])

                elif phase == "validation":
                    pbar.set_description(f'Epoch {epoch + 1}/{epochs} (validation round)')

                    optimizer.zero_grad()
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        if not value.grad == None:
                            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                    net.eval()

                    n_val = len(val_loader)  # the number of batch
                    dice_sum = 0
                    dice_sums = [0, 0, 0]
                    loss_sum = 0
                    accuracy_sum = 0
                    sensitivity_sum = 0
                    specificity_sum = 0
                    for batch in val_loader:
                        imgs, true_masks = batch['image'], batch['mask']
                        imgs = imgs.to(device=device, dtype=torch.float32)
                        true_masks = true_masks.to(device=device, dtype=mask_type)

                        print(f"True mask:\n   shape={true_masks.shape}\n    unique={torch.unique(true_masks)}")

                        with torch.no_grad():
                            masks_pred = net(imgs)

                        print(f"Pred mask:\n    shape={masks_pred.shape}\n    unique={torch.unique(masks_pred)}")


                        # Calculate dice score for each
                        if net.n_classes == 1:
                            dice_sum += dice_coeff((masks_pred > 0.5).float(), true_masks).item()
                        else:
                            probs = F.softmax(masks_pred, dim=1).data
                            max_idx = torch.argmax(probs, 0, keepdim=True)
                            one_hot = torch.FloatTensor(probs.shape).to(device=device)
                            one_hot.zero_()
                            one_hot.scatter_(0, max_idx, 1)

                            dice_sums[0] += dice_coeff(one_hot[:, 1, :, :], (true_masks == 1).float()).item() #[bg, c1, c2, c3] -> bg/c1 (0/1)
                            dice_sums[1] += dice_coeff(one_hot[:, 2, :, :], (true_masks == 2).float()).item()
                            dice_sums[2] += dice_coeff(one_hot[:, 3, :, :], (true_masks == 3).float()).item()

                        # Calculate validation loss
                        if net.n_classes > 1:
                            loss = criterion(masks_pred, true_masks.squeeze(1))
                        else:
                            loss = criterion(masks_pred, true_masks)
                        loss_sum += loss.item()

                        # Calculate accuracy, sensitivity and specificity scalars
                        """
                        probs = F.softmax(masks_pred, dim=1).data
                        max_idx = torch.argmax(probs, 0, keepdim=True)
                        one_hot = torch.FloatTensor(probs.shape).to(device=device)
                        one_hot.zero_()
                        one_hot.scatter_(0, max_idx, 1)
                        """

                        confusion_vector = one_hot / F.one_hot(true_masks.squeeze(1), num_classes=4).permute(0, 3, 1, 2) if net.n_classes > 1 else masks_pred / true_masks
                        # print(torch.unique(confusion_vector))
                        true_positives = torch.sum(confusion_vector == 1).item()
                        false_positives = torch.sum(confusion_vector == float('-inf')).item() + torch.sum(confusion_vector == float('inf')).item()
                        true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
                        false_negatives = torch.sum(confusion_vector == 0).item()

                        if (true_positives + true_negatives + false_negatives + false_positives) > 0:
                            accuracy_sum += (true_positives + true_negatives) / (
                            true_positives + true_negatives + false_positives + false_negatives)
                        if (true_positives + false_negatives) > 0:
                            sensitivity_sum = true_positives / (true_positives + false_negatives)
                        if (true_negatives + false_positives) > 0:
                            specificity_sum = true_negatives / (true_negatives + false_positives)


                        # Write a single image during validation
                        if (global_step % n_val) == 0:
                            writer.add_images('images', imgs, global_step)
                            if net.n_classes == 1:
                                writer.add_images('masks/true', true_masks, global_step)
                                writer.add_images('masks/pred', (masks_pred >= 0.5).float(), global_step)
                            else:
                                colors = [torch.Tensor([0., 0., 0.]), torch.Tensor([1., 0., 0.]), torch.Tensor([0., 1., 0.]), torch.Tensor([0., 0., 1.])]
                                batch, _, h, w = masks_pred.shape
                                pred_mask_img = torch.zeros((batch, h, w, 3))
                                true_mask_img = torch.zeros((batch, h, w, 3))
                                pred_idx = torch.argmax(masks_pred, dim=1)
                                for b in range(batch):
                                    for i in range(h):
                                        for j in range(w):
                                            pred_mask_img[b, i, j] = colors[pred_idx[b, i, j]]
                                            true_mask_img[b, i, j] = colors[true_masks.squeeze(1)[b, i, j]]
                                writer.add_images('masks/true', true_mask_img, global_step, dataformats='NHWC')
                                writer.add_images('masks/pred',  pred_mask_img, global_step, dataformats='NHWC')

                        global_step += global_step_size
                        pbar.update()

                    # write metrics
                    avg_loss = loss_sum / n_val
                    writer.add_scalar('Loss/validation', avg_loss, global_step)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    writer.add_scalar('metrics/accuracy', accuracy_sum / n_val, global_step)
                    writer.add_scalar('metrics/sensitivity', sensitivity_sum / n_val, global_step)
                    writer.add_scalar('metrics/specificity', specificity_sum / n_val, global_step)

                    if net.n_classes > 1:
                        writer.add_scalar('dice/tibia', dice_sums[0] / n_val, global_step)
                        writer.add_scalar('dice/femur_cart', dice_sums[1] / n_val, global_step)
                        writer.add_scalar('dice/tibia_cart', dice_sums[2] / n_val, global_step)

                    #Adjust learning rate based on metric
                    if net.n_classes == 1:
                        val_score = dice_sum / n_val
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('metrics/dice', val_score, global_step)
                    else:
                        val_score = avg_loss

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
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-r', '--schedule-factor', metavar='LRF', type=float, nargs='?', default=0.1,
                        help='Learning rate scheduler factor', dest='lrf')
    parser.add_argument('-p', '--schedule-patience', metavar='LRP', type=int, nargs='?', default=2,
                        help='Learning rate scheduler patience', dest='lrp')
    parser.add_argument('-o', '--optimizer-momentum', metavar='OM', type=float, nargs='?', default=0.9,
                        help='Optimizer momentum', dest='om')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
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
    n_classes = 1
    net = UNet(n_channels=1, n_classes=n_classes) #(background, tibia, femoral cartilage, tibial cartilage)
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

    lrs = [1e-3]
    try:
        for lr in lrs:
            net = UNet(n_channels=1, n_classes=n_classes)
            net.to(device=device)
            train_net(net=net,
                      epochs=args.epochs,
                      batch_size=args.batchsize,
                      lr=lr,
                      lrf=args.lrf,
                      lrp=args.lrp,
                      om=args.om,
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
