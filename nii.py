import os
import shutil
import numpy as np
import scipy.io as io
import nibabel as nib



shutil.rmtree('data_folder/', ignore_errors=True, onerror=None)
os.mkdir('data_folder/')
os.mkdir('data_folder/train/')
os.mkdir('data_folder/train/images')
os.mkdir('data_folder/train/labels')
os.mkdir('data_folder/test/')
os.mkdir('data_folder/test/images')
os.mkdir('data_folder/test/labels')
os.mkdir('data_folder/val/')
os.mkdir('data_folder/val/images')
os.mkdir('data_folder/val/labels')
print("Created folders")


i = 0
print("We are the knights who say...\n")
for f in os.listdir('ScanManTrim/'):
    mat = io.loadmat('ScanManTrim/'+f)
    scan = mat['scan']
    cart = np.maximum(mat['CartTM'], mat['CartFM'])
    
    nii_scan = nib.Nifti1Image(scan, affine=np.eye(4))
    nii_cart = nib.Nifti1Image(cart, affine=np.eye(4))
    
    if i < 10:
        nib.save(nii_scan, 'data_folder/train/images/image{}.nii'.format(i))
        print("Saved training image ",i)
        nib.save(nii_cart, 'data_folder/train/labels/image{}.nii'.format(i))
        print("Saved training label ",i)
    elif i < 15:
        nib.save(nii_scan, 'data_folder/test/images/image{}.nii'.format(i))
        print("Saved test image ",i)
        nib.save(nii_cart, 'data_folder/test/labels/image{}.nii'.format(i))
        print("Saved test label ",i)
    else:
        nib.save(nii_scan, 'data_folder/val/images/image{}.nii'.format(i))
        print("Saved validation image ",i)
        nib.save(nii_cart, 'data_folder/val/labels/image{}.nii'.format(i))
        print("Saved validation label ",i)
    i = i + 1
    

print("\nNII!")
