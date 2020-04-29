import numpy as np
import sys
import argparse
import logging
from unet.unet_model import UNet
from dice_loss import dice_coeff
from tqdm import tqdm
import torch
from torch.nn.functional import softmax
from utils.mri_dataset import MRI_Dataset
from torch.utils.data import DataLoader, SequentialSampler
import nibabel as nib

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

def predict(net, scan, mask=None):
    pass

def get_args():
    parser = argparse.ArgumentParser(description='Predict using a trained UNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')

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

    net = UNet(n_channels=1, n_classes=4)

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    dir_img = r"C:\Users\Niklas Magnussen\Desktop\TheBachelor\data_folder\test\images"
    dir_mask = r"C:\Users\Niklas Magnussen\Desktop\TheBachelor\data_folder\test\labels"

    dataset = MRI_Dataset(dir_img, dir_mask, net.n_classes, filter=False)

    sampler = SequentialSampler(dataset)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=sampler, num_workers=0, pin_memory=True, drop_last=False)

    mask_type = torch.float32 if net.n_classes == 1 else torch.long

    print("Creating predicted slices")
    n = 0
    N = len(dataset)
    predicted = []
    truths = []
    with tqdm(total=N, desc=f'Predictions ', unit='img') as pbar:
        for data in loader:
            img = data['image']
            img = img.to(device=device, dtype=torch.float32)
            true_mask = data['mask']
            true_mask = true_mask.to(device=device, dtype=mask_type)

            truths.append(true_mask)

            with torch.no_grad():
                pred_mask = net(img)

            probs = softmax(pred_mask, dim=1)
            """
                .data
            max_idx = torch.argmax(probs, 1, keepdim=True)
            one_hot = torch.FloatTensor(probs.shape).to(device=device)
            one_hot.zero_()
            one_hot.scatter_(1, max_idx, 1)
            """

            predicted.append(probs)
            n += 1
            pbar.update(1)

    predicted = np.array(predicted)
    print(predicted.shape)



    img_count = 0
    with tqdm(total=np.ceil(len(predicted) / dataset.image_dims[0]), desc=f'Segmentation volumes', unit='segmentations') as pbar:
        for id in dataset.ids:
            """
            [    first view    ][  second view  ][  third view  ][ next scan
             0, 1, 2, ... , 169, 170,  ... , 339, 340,  ..., 509, 510,
            """
            i = img_count

            true_mask = torch.cat(truths[i:i + dataset.image_dims[0]])
            print(true_mask.shape)

            #print(f"Volume 1: {i} : {i + dataset.image_dims[0]}")
            volume1 = slices_to_volume(predicted[i:i + dataset.image_dims[0]])
            volume_to_nii(volume1, "pred1" + id)

            i += dataset.image_dims[0]
            pbar.update(1)

            #print(f"Volume 2: {i} : {i + dataset.image_dims[1]}")
            # permute rotates the volume image to match the ground truth label
            volume2 = slices_to_volume(predicted[i:i + dataset.image_dims[1]]).permute(2, 1, 0, 3)
            volume_to_nii(volume2, "pred2" + id)

            i += dataset.image_dims[1]
            pbar.update(1)

            #print(f"Volume 3: {i} : {i + dataset.image_dims[2]}")
            volume3 = slices_to_volume(predicted[i:i + dataset.image_dims[2]]).permute(2, 1, 3, 0)
            i += dataset.image_dims[2]
            volume_to_nii(volume3, "pred3" + id)
            pbar.update(1)

            avg_volume = (volume1 + volume2 + volume3) / 3.0
            print("dice scores:\n  tibia: {}\n  femoral cartilage: {}\n  tibial cartilage: {}".format(dice(avg_volume, true_mask, 1), dice(avg_volume, true_mask, 2), dice(avg_volume, true_mask, 3)))
            volume_to_nii(avg_volume, "avgpred" + id)

            img_count += i

    """
    Compute Dice volume overlap on the three different views.
    Combine the three view segmentations to a single volume, that is the average of the three.
    Save predicted volume as .nii file
    
    Problemer: 
    - Tjek at volumes indeholder de rigtige værdier (ikke er tomme)
    - Hvordan laver vi (vægtet?) gennemsnit med class values og ikke one hot?
    - Hvordan håndterer vi (un)padding. Ground truth labels er paddede, det er predictions ikke.
    - Kun den mediale del af knæet har ground truth brusk. Hvis prediction mask gætter brusk i den anden halvdel, vil det tælle negativt i dice score, selvom det kan være rigtigt.
    - 
    """

