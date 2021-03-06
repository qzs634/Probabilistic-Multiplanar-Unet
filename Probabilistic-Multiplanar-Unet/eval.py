import numpy as np
import os
import gc
import argparse
import logging
from dice_loss import dice_coeff
from tqdm import tqdm
import torch
from torch.nn.functional import softmax
from utils.mri_dataset import MRI_Dataset
from torch.utils.data import DataLoader, SequentialSampler
from trainer import UNetTrainer, ProbUNetTrainer
import nibabel as nib

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
    parser.add_argument('-m', '--model', dest='net', type=str, default="unet",
                        help='what model to use: unet or probunet')

    return parser.parse_args()

# Calculate dice score
    # Find index in class vector for prediction with largest value
    # Convert prediction to a one hot vector
    # Compare prediction to ground truth
def dice(pred, truth, class_index):
    max_idx = torch.argmax(pred, 1, keepdim=True)
    one_hot = torch.FloatTensor(pred.shape).to(device=device)
    one_hot.zero_()
    one_hot.scatter_(1, max_idx, 1)

    dice = dice_coeff(one_hot[:, class_index, :, :], (truth == class_index).float().squeeze(1)).item()
    return dice

def volume_to_nii(volume, title, predicted=True):
    argmax = torch.argmax(volume, axis=1)
    if predicted:
        nii = nib.Nifti1Image(argmax.cpu().numpy().astype(np.float32), affine=np.eye(4))
    else:
        nii = nib.Nifti1Image(volume.cpu().numpy().astype(np.float32), affine=np.eye(4))
    nib.save(nii, title)

# Recombine slices from a volume along view
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

    if args.net == "unet":
        train = UNetTrainer(device, n_channels=1, n_classes=3, load_model=args.load)
    elif args.net == "probunet":
        train = ProbUNetTrainer(device, n_channels=1, n_classes=3, load_model=args.load, latent_dim=6, beta=10)
    else:
        print("Error! {} is not a valid model".format(args.net))

    # Image locations
    dir_img = r"C:\Users\Niklas Magnussen\Desktop\TheBachelor\data_folder\test\images"
    dir_mask = r"C:\Users\Niklas Magnussen\Desktop\TheBachelor\data_folder\test\labels"

    if args.dir is not None:
        dir_img = os.path.join(args.dir, "images")
        dir_mask = os.path.join(args.dir, "labels")

    dataset = MRI_Dataset(dir_img, dir_mask, train.net.n_classes, filter=False)

    # The slices has to be loaded in sequential order.
    # Its important when recombining the slices.
    sampler = SequentialSampler(dataset)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=sampler, num_workers=0, pin_memory=True, drop_last=False)

    mask_type = torch.float32 if train.net.n_classes == 1 else torch.long

    print("Creating predicted slices")
    n = 0
    N = len(dataset)

    # number of slices in a volume
    n_slices = dataset.image_dims[0] + dataset.image_dims[1] + dataset.image_dims[2]

    predicted = []
    truths = []

    dice_sums = [] # np.array([0.0] * (train.net.n_classes - 1))
    best_dice = None

    slice_count = 0
    img_count = 0

    # Dice scores for each view.
    # Saved as a array in order to calculate mean and standard deviation
    vol_1_dice = [] #np.array([0.0] * (train.net.n_classes - 1))
    vol_2_dice = [] #np.array([0.0] * (train.net.n_classes - 1))
    vol_3_dice = [] #np.array([0.0] * (train.net.n_classes - 1))

    with tqdm(total=N, desc=f'Predictions ', unit='img') as pbar:
        for data in loader:

            # Move data to GPU
            img = data['image']
            img = img.to(device=device, dtype=torch.float32)
            true_mask = true_mask.to(device=device, dtype=mask_t
            true_mask = data['mask']ype)

            truths.append(true_mask)

            if args.net == "unet":
                with torch.no_grad():
                    pred_masks = train.predict(img, true_mask)
            elif args.net == "probunet":
                pred_masks = None  # 6, c, h, w
                with torch.no_grad():
                    for _ in range(5):
                        if pred_masks is None:
                            pred_masks = train.predict(img, true_mask)
                        else:
                            pred_masks + train.predict(img, true_mask)

                pred_masks /= 5

            # Convert prediction results to a probability distribution
            probs = softmax(pred_masks, dim=1)

            predicted.append(probs)
            slice_count += 1

            if slice_count == n_slices:

                """
                slice count is equal to all slices in volume
                """

                i = 0
                predicted = np.array(predicted)

                # Create ground truth volume
                true_mask = torch.cat(truths[i:i + dataset.image_dims[0]])
                true_mask = slices_to_volume(true_mask)

                # Create first views volume and save its dice scores
                volume1 = slices_to_volume(predicted[i:i + dataset.image_dims[0]]) # [1, 3, 256, 256, 256]
                vol_1_dice.append(np.array([dice(volume1, true_mask, 1), dice(volume1, true_mask, 2)]))
                i += dataset.image_dims[0]

                # Create second views volume and save its dices scores
                # Permute rotates the volume image to match the ground truth label
                volume2 = slices_to_volume(predicted[i:i + dataset.image_dims[1]]).permute(2, 1, 0, 3)
                vol_2_dice.append(np.array([dice(volume2, true_mask, 1), dice(volume2, true_mask, 2)]))
                i += dataset.image_dims[1]

                # Create third views volume and save its dices scores
                # Permute rotates the volume image to match the ground truth label
                volume3 = slices_to_volume(predicted[i:i + dataset.image_dims[2]]).permute(2, 1, 3, 0)
                vol_3_dice.append(np.array([dice(volume3, true_mask, 1), dice(volume3, true_mask, 2)]))
                i += dataset.image_dims[2]

                # Take an average of all the views and save the volume as a NII file
                avg_volume = (volume1 + volume2 + volume3) / 3.0
                volume_to_nii(avg_volume, os.path.join(r"C:\Users\Niklas Magnussen\Desktop\TheBachelor\Pytorch-UNet-modified\predictions\probunet-labels", dataset.ids[img_count]))

                # Clear GPU memory
                del volume1
                del volume2
                del volume3
                torch.cuda.empty_cache()

                # Calculate and average dice score for each class, for the whole volume
                dices = np.array([dice(avg_volume, true_mask, 1), dice(avg_volume, true_mask, 2)])

                dice_sums.append(dices)
                del avg_volume

                img_count += 1
                slice_count = 0
                predicted = []
                truths = []

                gc.collect()

            n += 1
            pbar.update(1)

        print("best dice: ", best_dice)
        v1_mean = np.mean(np.array(vol_1_dice), axis=0)
        v1_std = np.std(np.array(vol_1_dice), axis=0)
        print(f"view 1 dice: mean={v1_mean}, std={v1_std}")

        v2_mean = np.mean(np.array(vol_2_dice), axis=0)
        v2_std = np.std(np.array(vol_2_dice), axis=0)
        print(f"view 2 dice: mean={v2_mean}, std={v2_std}")

        v3_mean = np.mean(np.array(vol_3_dice), axis=0)
        v3_std = np.std(np.array(vol_3_dice), axis=0)
        print(f"view 3 dice: mean={v3_mean}, std={v3_std}")

        avg_mean = np.mean(np.array(dice_sums), axis=0)
        avg_std = np.std(np.array(dice_sums), axis=0)
        print(f"avg volume: mean={avg_mean}, std={avg_std}")