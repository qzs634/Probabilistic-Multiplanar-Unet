import numpy as np
import sys
import os
import argparse
import logging
from dice_loss import dice_coeff
from tqdm import tqdm
import torch
from torch.nn.functional import softmax
from utils.mri_dataset import MRI_Dataset
from torch.utils.data import DataLoader, SequentialSampler
import nibabel as nib
import matplotlib.pyplot as plt
from trainer import UNetTrainer, ProbUNetTrainer


torch.set_printoptions(threshold=5000)
#np.set_printoptions(threshold=sys.maxsize)

"""
Eval:
- load net and state_dict
- load scan and optional ground truth label
- predict labels along all views (3 standard axis)
- if optional label:
    - compare all view's segmentation volumes against ground truth
- combine all segmentations to one average segmentation volume
"""
def get_args():
    parser = argparse.ArgumentParser(description='Predict using a trained UNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-d', '--dir', dest='dir', type=str, default=None,
                        help='image and label superdirs.')

    return parser.parse_args()

def dice(pred, truth, class_index):
    max_idx = torch.argmax(pred, 1, keepdim=True)
    one_hot = torch.FloatTensor(pred.shape).to(device=device)
    one_hot.zero_()
    one_hot.scatter_(1, max_idx, 1)

    dice = dice_coeff(one_hot[:, class_index, :, :], (truth == class_index).float().squeeze(1)).item()
    return dice

def volume_to_nii(volume, title):
    argmax = torch.argmax(volume, axis=1)
    nii = nib.Nifti1Image(argmax.cpu().numpy().astype(np.float32), affine=np.eye(4))
    nib.save(nii, title)

def slices_to_volume(slices):
    volume = slices[0]
    for slice in slices[1:]:
        volume = torch.cat([volume,slice])

    return volume

if __name__ == "__main__":
    printstr = """
 /$$   /$$ /$$   /$$ /$$$$$$$$ /$$$$$$$$       /$$$$$$$$ /$$    /$$  /$$$$$$  /$$       /$$   /$$  /$$$$$$  /$$$$$$$$ /$$$$$$  /$$$$$$  /$$   /$$
| $$  | $$| $$$ | $$| $$_____/|__  $$__/      | $$_____/| $$   | $$ /$$__  $$| $$      | $$  | $$ /$$__  $$|__  $$__/|_  $$_/ /$$__  $$| $$$ | $$
| $$  | $$| $$$$| $$| $$         | $$         | $$      | $$   | $$| $$  \ $$| $$      | $$  | $$| $$  \ $$   | $$     | $$  | $$  \ $$| $$$$| $$
| $$  | $$| $$ $$ $$| $$$$$      | $$         | $$$$$   |  $$ / $$/| $$$$$$$$| $$      | $$  | $$| $$$$$$$$   | $$     | $$  | $$  | $$| $$ $$ $$
| $$  | $$| $$  $$$$| $$__/      | $$         | $$__/    \  $$ $$/ | $$__  $$| $$      | $$  | $$| $$__  $$   | $$     | $$  | $$  | $$| $$  $$$$
| $$  | $$| $$\  $$$| $$         | $$         | $$        \  $$$/  | $$  | $$| $$      | $$  | $$| $$  | $$   | $$     | $$  | $$  | $$| $$\  $$$
|  $$$$$$/| $$ \  $$| $$$$$$$$   | $$         | $$$$$$$$   \  $/   | $$  | $$| $$$$$$$$|  $$$$$$/| $$  | $$   | $$    /$$$$$$|  $$$$$$/| $$ \  $$
 \______/ |__/  \__/|________/   |__/         |________/    \_/    |__/  |__/|________/ \______/ |__/  |__/   |__/   |______/ \______/ |__/  \__/


                                                                                                                                                 """
    print(printstr)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    train = UNetTrainer(device, n_channels=1, n_classes=3, load_model=args.load)

    dir_img = r"C:\Users\Niklas Magnussen\Desktop\TheBachelor\data_folder\test\images"
    dir_mask = r"C:\Users\Niklas Magnussen\Desktop\TheBachelor\data_folder\test\labels"

    if args.dir is not None:
        dir_img = os.path.join(args.dir, "images")
        dir_mask = os.path.join(args.dir, "labels")


    dataset = MRI_Dataset(dir_img, dir_mask, train.net.n_classes, filter=True)

    sampler = SequentialSampler(dataset)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=sampler, num_workers=8, pin_memory=True, drop_last=False)

    mask_type = torch.float32 if train.net.n_classes == 1 else torch.long

    print("Creating predicted slices")
    n = 0
    N = len(dataset)
    predicted = []
    truths = []
    dice_sums = np.array([0.0] * (train.net.n_classes - 1))
    dices = 0
    best_batch = None
    best_dice = None
    train.net.eval()
    with tqdm(total=N, desc=f'Predictions ', unit='img') as pbar:
        for data in loader:
            img = data['image']
            img = img.to(device=device, dtype=torch.float32)
            true_mask = data['mask']
            true_mask = true_mask.to(device=device, dtype=mask_type)

            #print(torch.unique(true_mask).cpu().numpy())

            with torch.no_grad():
                pred_mask = train.predict(img, true_mask)

            dice = train.eval(img, true_mask, pred_mask)
            if train.net.n_classes > 1:
                dice_sums += dice
                if best_dice is None or (((dice_sums[0] + dice_sums[1]) / 2) > ((best_dice[0] + best_dice[1]) / 2)) and all(elem in list(torch.unique(true_mask).cpu().numpy()) for elem in [1, 2]):
                    best_batch = data
                    best_dice = dice
            else:
                dices += dice[0]
                if best_dice is None or dice[0] > best_dice and torch.max(true_mask > 0):
                    best_dice = dice[0]
                    best_batch = data
            pbar.update(1)

    if train.net.n_classes > 1:
        print(dice_sums / N)
    else:
        print(dices / N)

    print(best_dice)

    img=best_batch['image']
    img = img.to(device=device, dtype=torch.float32)
    true_mask=best_batch['mask']
    true_mask = true_mask.to(device=device, dtype=mask_type)

    with torch.no_grad():
        pred_mask = train.predict(img, true_mask)

    mask_img = train.mask_to_image(true_mask)
    mask_img = (mask_img.cpu().numpy().squeeze() * 255).astype(np.uint8)

    pred_mask_img = train.mask_to_image(pred_mask, prediction=True)
    pred_mask_img = (pred_mask_img.cpu().numpy().squeeze() * 255).astype(np.uint8)

    if train.net.n_classes > 1:
        mask_img = np.transpose(mask_img, (1, 2, 0))
        pred_mask_img = np.transpose(pred_mask_img, (1, 2, 0))

    fig, ax = plt.subplots(1,3)
    ax[0].imshow((img.cpu().numpy().squeeze() * 255).astype(np.uint8), cmap='Greys_r')
    ax[0].set_title("(a)")
    ax[1].imshow(mask_img, cmap='Greys_r')
    ax[1].set_title("(b)")
    ax[2].imshow(pred_mask_img, cmap='Greys_r')
    ax[2].set_title("(c)")

    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')

    fig.savefig("uneteval.png", dpi=300, bbox_inches='tight')
    plt.show()

