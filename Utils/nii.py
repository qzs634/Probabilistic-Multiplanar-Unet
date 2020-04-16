import os
import sys
import shutil
import numpy as np
import scipy.io as io
import nibabel as nib


def crop3d(mat):
    scan = mat['scan']
    cart_tm = mat['CartTM']
    cart_fm = mat['CartFM']
    tibia = mat['Tibia']
    rows, cols, slices = mat['scan'].shape
    isLeft = not bool(mat['isright'])

    # if right leg
    start_index = rows - 1
    end_index = 0
    step = -1
    if isLeft:
        # start fra cols-1 -- til 0
        start_index = 0
        end_index = rows - 1
        step = 1

    label_img = np.maximum(cart_tm, cart_fm)
    for i in range(start_index, end_index, step):
        if np.max(label_img[i, :, :]) > 0:
            scan = scan[end_index:i, :, :]
            cart_tm = cart_tm[end_index:i, :, :]
            cart_fm = cart_fm[end_index:i, :, :]
            tibia = tibia[end_index:i, :, :]
            break

    return (cart_tm, cart_fm, tibia, scan)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "ScanManTrim"
    shutil.rmtree(os.path.join('data_folder'), ignore_errors=True, onerror=None)
    os.mkdir(os.path.join('data_folder'))
    os.mkdir(os.path.join('data_folder', 'train'))
    os.mkdir(os.path.join('data_folder', 'train', 'images'))
    os.mkdir(os.path.join('data_folder', 'train', 'labels'))
    os.mkdir(os.path.join('data_folder', 'test'))
    os.mkdir(os.path.join('data_folder', 'test', 'images'))
    os.mkdir(os.path.join('data_folder', 'test', 'labels'))
    os.mkdir(os.path.join('data_folder', 'val'))
    os.mkdir(os.path.join('data_folder', 'val', 'images'))
    os.mkdir(os.path.join('data_folder', 'val', 'labels'))
    print("Created folders")


    i = 0
    print("We are the knights who say...\n")
    for f in os.listdir(path):
        file_path = os.path.join(path, f)
        print(file_path)
        mat = io.loadmat(os.path.join(path, f))
        scan = mat['scan']
        cart_tm = mat['CartTM'] * 3
        cart_fm = mat['CartFM'] * 2
        tibia = mat['Tibia']

        cart = np.maximum(np.maximum(cart_tm, cart_fm), tibia)

        nii_scan = nib.Nifti1Image(scan, affine=np.eye(4))
        nii_cart = nib.Nifti1Image(cart, affine=np.eye(4))

        if i < 1000:
            nib.save(nii_scan, os.path.join('data_folder','train', 'images', 'image{}.nii'.format(i)))
            print("Saved training image ",i)
            nib.save(nii_cart, os.path.join('data_folder', 'train', 'labels', 'image{}.nii'.format(i)))
            print("Saved training label ",i)
        elif i < 15:
            nib.save(nii_scan, os.path.join('data_folder', 'test', 'images', 'image{}.nii'.format(i)))
            print("Saved test image ",i)
            nib.save(nii_cart, os.path.join('data_folder', 'test', 'labels', 'image{}.nii'.format(i)))
            print("Saved test label ",i)
        else:
            nib.save(nii_scan, os.path.join('data_folder', 'val', 'images', 'image{}.nii'.format(i)))
            print("Saved validation image ",i)
            nib.save(nii_cart, os.path.join('data_folder', 'val', 'labels', 'image{}.nii'.format(i)))
            print("Saved validation label ",i)
        i = i + 1


    print("\nNII!")
