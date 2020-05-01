from .trainer import Trainer
import numpy as np
from model import UNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from dice_loss import dice_coeff

class UNetTrainer(Trainer):

    def __init__(self, device, n_channels=1, n_classes=1, load_model=None):
        self.device = device
        self.mask_type = torch.float32 if n_classes == 1 else torch.long
        net = UNet(n_channels=n_channels, n_classes=n_classes)

        if load_model is not None:
            self.net.load_state_dict(
                torch.load(load_model, map_location=device)
            )

        self.net = net.to(device)
        self.criterion = nn.BCELoss() if self.net.n_classes == 1 else nn.CrossEntropyLoss()

    def predict(self, imgs, true_masks):
        masks_pred = self.net(imgs)

        return masks_pred

    def loss(self, imgs, true_masks, masks_pred):

        if self.net.n_classes > 1:
            loss = self.criterion(masks_pred, true_masks.squeeze(1))
        else:
            loss = self.criterion(masks_pred, true_masks)

        return loss

    def eval(self, imgs, true_masks, masks_pred):

        # Calculate dice score for each
        if self.net.n_classes == 1:
            dice = [dice_coeff((masks_pred > 0.5).float(), true_masks).item()]
        else:
            dice = []
            probs = F.softmax(masks_pred, dim=1).data
            max_idx = torch.argmax(probs, 1, keepdim=True)
            one_hot = torch.FloatTensor(probs.shape).to(device=self.device)
            one_hot.zero_()
            one_hot.scatter_(1, max_idx, 1)

            dice.append(dice_coeff(one_hot[:, 1, :, :], (true_masks == 1).float().squeeze(1)).item())
            dice.append(dice_coeff(one_hot[:, 2, :, :], (true_masks == 1).float().squeeze(1)).item())
            dice.append(dice_coeff(one_hot[:, 3, :, :], (true_masks == 1).float().squeeze(1)).item())


        # Calculate accuracy, sensitivity and specificity scalars

        """
        probs = F.softmax(masks_pred, dim=1).data
        max_idx = torch.argmax(probs, 0, keepdim=True)
        one_hot = torch.FloatTensor(probs.shape).to(device=device, dtype=mask_type)
        one_hot.zero_()
        one_hot.scatter_(1, max_idx, 1)


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
        """

        return np.array(dice)

    def mask_to_image(self, masks, prediction=False):
        if self.net.n_classes == 1:
            if prediction:
                img = (masks >= 0.5).float()
            else:
                img = masks
        else:
            # TODO: Extend the colors if more classes are added
            colors = [torch.Tensor([0., 0., 0.]), torch.Tensor([1., 0., 0.]), torch.Tensor([0., 1., 0.]),
                      torch.Tensor([0., 0., 1.])]
            batch, _, h, w = masks.shape
            if prediction:
                pred_mask_img = torch.zeros((batch, h, w, 3))
                pred_idx = torch.argmax(masks, dim=1)
                for b in range(batch):
                    for i in range(h):
                        for j in range(w):
                            pred_mask_img[b, i, j] = colors[pred_idx[b, i, j]]

                img = pred_mask_img.permute(0, 3, 1, 2)
            else:
                true_mask_img = torch.zeros((batch, h, w, 3))
                for b in range(batch):
                    for i in range(h):
                        for j in range(w):
                            true_mask_img[b, i, j] = colors[masks.squeeze(1)[b, i, j]]

                img = true_mask_img.permute(0, 3, 1, 2)

            return img