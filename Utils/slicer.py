#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d
#import time
import nibabel as nib
matplotlib.rcParams['figure.figsize'] = [15, 10]

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


def crop3d(mat):
    scan = mat['scan']
    cart_tm = mat['CartTM']
    cart_fm = mat['CartFM']
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
        if np.max(label_img[i,:,:]) > 0:
            scan =       scan[end_index:i, :, :]
            cart_tm = cart_tm[end_index:i, :, :]
            cart_fm = cart_fm[end_index:i, :, :]
            break

    return (cart_tm, cart_fm, scan)




            
                
            

#filenames = [f for f in os.listdir('ScanManTrim')]
#f = [scipy.io.loadmat('ScanManTrim/'+f) for f in filenames]
if len(sys.argv) > 1:
        file_index = int(sys.argv[1])
else:
        file_index = 0

nii_im =  nib.load(r'C:\Users\Niklas Magnussen\Desktop\TheBachelor\data_folder(Binary)\train\images\image15.nii').get_fdata()
nii_lab = nib.load(r'C:\Users\Niklas Magnussen\Desktop\TheBachelor\data_folder(Binary)\train\labels\image15.nii').get_fdata()

#mat = f[file_index]
#crop3d(mat)
#scan = mat['scan']
#scan_norm = (scan / np.max(scan)) * 255
#cart_tm = mat['CartTM'] * 255
#cart_fm = mat['CartFM'] * 255
#tibia = mat['Tibia'] * 255
#k_img = np.stack([cart_tm, cart_fm, tibia],3)



index = 0
def multi_slice_viewer(X,Y):
    global index
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots(1,2)
    ax[0].set_title("predicted")
    ax[0].set_ylabel("slice " + str(index))
    ax[1].set_title("ground truth")
    ax[0].volume = X
    ax[1].volume = Y
    ax[0].im = ax[0].imshow(X[index,:,:], cmap='Greys_r')
    ax[1].im = ax[1].imshow(Y[index,:,:], cmap='Greys_r')
    ax[1].im.set_clim(0,1)
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    global index
    x = ax[0].volume
    y = ax[1].volume
    index = (index - 1) % x.shape[0]  # wrap around using %
    ax[0].set_ylabel("slice " + str(index))
    ax[0].im.set_data(x[index,:,:])
    ax[1].im.set_data(y[index,:,:])

def next_slice(ax):
    global index
    x = ax[0].volume
    y = ax[1].volume
    index = (index + 1) % x.shape[0]  # wrap around using %
    ax[0].set_ylabel("slice " + str(index)) 
    ax[0].im.set_data(x[index,:,:])
    ax[1].im.set_data(y[index,:,:])

# side view - sagital plane (0, 1, 2)
# front view - coronal plane (1, 0, 2)
# top-down view - axial plane (2, 1, 0)
#tm, fm, crop_scan = crop3d(mat)
#cart = np.maximum(tm, fm)
#multi_slice_viewer(nii_im.transpose(1, 0, 2), nii_lab.transpose(1, 0, 2))
fig, ax = plt.subplots(1, 3)
ax[0].imshow(nii_im[nii_im.shape[0] // 2], cmap='Greys_r')
plt.imsave("saggital.png", nii_im[25], cmap='Greys_r')
ax[1].imshow(nii_im.transpose(1, 0, 2)[nii_im.shape[1] // 2], cmap='Greys_r')
plt.imsave("coronal.png", nii_im.transpose(1, 0, 2)[nii_im.shape[1] // 2], cmap='Greys_r')
ax[2].imshow(nii_im.transpose(2, 1, 0)[nii_im.shape[2] // 2], cmap='Greys_r')
plt.imsave("axial.png", nii_im.transpose(2, 1, 0)[nii_im.shape[2] // 2], cmap='Greys_r')
plt.show()

