from utils.dataset import BasicDataset
import numpy as np
from os import listdir, path
from glob import glob
import torch
from torch.utils.data import Dataset, SubsetRandomSampler
from torch.utils.data.dataloader import default_collate
import nibabel as nib
import logging

class MRI_Dataset(Dataset):

    def __init__(self, imgs_dir, masks_dir, n_classes, filter=True, ):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.n_classes = n_classes
        self.len = 0
        self.views = self.initialize_views(use_standard_axis=True)

        # Array of filenames
        self.ids = listdir(imgs_dir)#[path.splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]

        logging.info(f'Creating index mapping.')
        idx = self.ids[0]

        # Get size of images.
        # Largest image dimension is the baseline. Images smaller than this will be padded.
        img = nib.load(path.join(self.imgs_dir, idx)).get_fdata()
        self.image_dims = tuple([np.max(img.shape)] * len(img.shape))

        # Index mapping:
            # Loop through all files in ids.
            # Loop through all views
            # Loop through all slices in the view
            # Map specific slice from specific view and file to a flat array.
            # Filter all images out containing only background
        self.index_map = []
        for scan in range(len(self.ids)):
            mask = self.pad_dimensions(nib.load(path.join(self.masks_dir, self.ids[scan])).get_fdata())
            for view in range(len(self.views)):
                dim_shape = mask.shape[view]
                for slice in range(dim_shape):
                    mask_slice = self.sample_slice(mask, self.views[view], slice)

                    if filter:
                        if np.max(mask_slice) > 0:
                            self.index_map.append((scan, view, slice))
                    else:
                        self.index_map.append((scan, view, slice))

        self.len = len(self.index_map)
        logging.info(f'Creating dataset of {len(self.ids)} scans, and {self.len} slices')


    # Number of slices
    def __len__(self):
        return self.len

    # Retrieve standard axes
    def initialize_views(self, use_standard_axis=False):
        standard_axis = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

        if use_standard_axis:
            views = standard_axis

        return views

    # Given an image, view and slice and.
    # Retrieves image slice from that index.
    def sample_slice(self, image, view, slice_index):
        # TODO generalize this instead of hardcoded slicing.
        if np.array_equal(view, self.views[0]): # [1, 0, 0] = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
            image_slice = image[slice_index, :, :]
        elif np.array_equal(view, self.views[1]):
            image_slice = image[:, slice_index, :]
        elif np.array_equal(view, self.views[2]):
            image_slice = image[:, :, slice_index]
        else:
            print("No valid view")
            exit()

        return image_slice

    # Pad images to same size
    def pad_dimensions(self, image):
        dim_diff = np.max(image.shape) - np.min(image.shape)

        if dim_diff != 0:
            if np.argmin(image.shape) == 0:
                # pad rows
                image = np.concatenate((image, np.zeros((dim_diff, image.shape[1], image.shape[2]))), axis=0)
            elif np.argmin(image.shape) == 1:
                # pad columns
                image = np.concatenate((image, np.zeros((image.shape[0], dim_diff, image.shape[2]))), axis=1)
            else:
                image = np.concatenate((image, np.zeros((image.shape[0], image.shape[1], dim_diff))), axis=2)

        return image

    # Restructure image arrays to fit Pytorch
    @classmethod
    def preprocess(cls, img, label=False):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        # HWC to CHW
        img_trans = np.transpose(img, [2, 0, 1])
        if not label:
            if not np.max(img_trans) == 0:
                img_trans = img_trans / np.max(img_trans)

        return img_trans

    # The reason for the index mapping, is so that the getitem function, can utilize it.
    # Given the index map, unpacks the specific scan, view and slice.
    # Returns image slice and label slice.
    def __getitem__(self, i):
        # i -> image index, view index, slice index
        # totale slices = 24 x 104 x 170 x 170
        scan_i, view_i, slice_i = self.index_map[i]
        idx = self.ids[scan_i]


        img = nib.load(path.join(self.imgs_dir,  idx)).get_fdata()
        img = self.pad_dimensions(img)
        mask = nib.load(path.join(self.masks_dir,  idx)).get_fdata()
        mask = self.pad_dimensions(mask)

        assert img.shape == mask.shape, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        view = self.views[view_i]

        img_slice = self.sample_slice(img, view, slice_i)
        mask_slice = self.sample_slice(mask, view, slice_i)

        img = self.preprocess(img_slice, label=False)
        mask = self.preprocess(mask_slice, label=True)


        if True: #np.max(mask) > 1:
            return {'image': torch.from_numpy(img).float(), 'mask': torch.from_numpy(mask).float()}

        return None