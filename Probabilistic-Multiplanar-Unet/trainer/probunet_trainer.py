from .trainer import Trainer
import numpy as np
from model import ProbabilisticUnet
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.probabilistic_unet.utils import l2_regularisation
from dice_loss import dice_coeff

class ProbUNetTrainer(Trainer):

    def __init__(self, device, n_channels=1, n_classes=1, load_model=None, latent_dim=6, beta=10):
        self.device = device
        self.mask_type = torch.float32
        self.name = "probunet"
        self.net = ProbabilisticUnet(input_channels=n_channels, num_classes=n_classes, num_filters=[64,128,256,512,1024], latent_dim=latent_dim, no_convs_fcomb=4, beta=beta)

        if load_model is not None:

            self.net.load_state_dict(
                torch.load(load_model, map_location=device), strict=False
            )

        self.net = self.net.to(device)
        self.criterion = nn.BCELoss() if self.net.n_classes == 1 else nn.CrossEntropyLoss()

    def predict(self, imgs, true_masks, z=None):
        train = torch.is_grad_enabled()
        self.net.forward(imgs, true_masks, training=train)
        masks_pred = self.net.sample(testing=not train) if z is None else self.net.sample_at(z)

        return masks_pred

    def loss(self, imgs, true_masks, masks_pred):
        elbo = self.net.elbo(true_masks)
        #reg_loss = l2_regularisation(self.net.posterior) + l2_regularisation(self.net.prior) + l2_regularisation(self.net.fcomb.layers)
        loss = -elbo# + 1e-5 * reg_loss

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

            for k in range(1, one_hot.shape[1]):
                input = one_hot[:, k, :, :]
                target = (true_masks == k).float().squeeze(1)
                d = dice_coeff(input, target)
                dice.append(d.item())

        return np.array(dice)

    def mask_to_image(self, masks, prediction=False):
        if self.net.n_classes == 1:
            if prediction:
                img = (masks >= 0.5).float()
            else:
                img = masks
        else:
            # TODO: Extend the colors if more classes are added
            colors = [torch.Tensor([0., 0., 0.]), torch.Tensor([0., 0., 1.]),
                      torch.Tensor([0., 1., 0.]), torch.Tensor([1., 0., 0.])]
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
                            index = int(masks.squeeze(1)[b, i, j])
                            true_mask_img[b, i, j] = colors[index]

                img = true_mask_img.permute(0, 3, 1, 2)

        return img