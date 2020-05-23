import os
import sys
import shutil
import numpy as np
import scipy.io as io
import hdf5storage as hdf5
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
    print("Created folders")


    i = 0
    print("We are the knights who say...\n")
    n = len(os.listdir(path))
    print(f"Saving {n} scans.")
    for f in os.listdir(path):
        file_path = os.path.join(path, f)
        mat = hdf5.loadmat(os.path.join(path, f))
        #print(mat)
        scan = mat['scan']
        """
        # 'FemoralCartilage', 'LateralFemoralCartilage', 'LateralMeniscus', 'LateralTibialCartilage', 'MedialFemoralCartilage', 'MedialMeniscus', 'MedialTibialCartilage', 'PatellarCartilage', 'Tibia'
        fcl = mat['LateralFemoralCartilage'].astype(int) * 8
        fcm = mat['MedialFemoralCartilage'].astype(int) * 7
        mm = mat['MedialMeniscus'].astype(int) * 6
        lm = mat['LateralMeniscus'].astype(int) * 5
        tcl = mat['LateralTibialCartilage'].astype(int) * 4
        tcm = mat['MedialTibialCartilage'].astype(int) * 3
        pc = mat['PatellarCartilage'].astype(int) * 2
        tib = mat['Tibia'].astype(int)

        cart = np.maximum(fcl, fcm)
        cart = np.maximum(cart, mm)
        cart = np.maximum(cart, lm)
        cart = np.maximum(cart, tcl)
        cart = np.maximum(cart, tcm)
        cart = np.maximum(cart, pc)
        cart = np.maximum(cart, tib)
        """

        # Binary segmentations
        cart_tm = mat['CartTM'].astype(int)
        cart_fm = mat['CartFM'].astype(int) * 2

        # Multiclass segmentations
        # cart_tm = mat['CartTM'] * 2
        # cart_fm = mat['CartFM'] * 1

        cart = np.maximum(cart_tm, cart_fm)


        nii_scan = nib.Nifti1Image(scan, affine=np.eye(4))
        nii_cart = nib.Nifti1Image(cart, affine=np.eye(4))

        if i < int(n * 0.85):
            nib.save(nii_scan, os.path.join('data_folder','train', 'images', 'image{}.nii'.format(i)))
            print("Saved training image ",i)
            nib.save(nii_cart, os.path.join('data_folder', 'train', 'labels', 'image{}.nii'.format(i)))
            print("Saved training label ",i)
        else:
            nib.save(nii_scan, os.path.join('data_folder', 'test', 'images', 'image{}.nii'.format(i)))
            print("Saved test image ",i)
            nib.save(nii_cart, os.path.join('data_folder', 'test', 'labels', 'image{}.nii'.format(i)))
            print("Saved test label ",i)
        i = i + 1


    print("\nNII!")
