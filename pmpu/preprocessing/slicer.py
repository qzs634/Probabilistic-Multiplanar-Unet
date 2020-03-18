import os
from images.image_loader  import ImageLoader
from images.image_label_pair  import ImageLabelPair
import random
import numpy as np
import matplotlib.pyplot as plt



class Slicer():
    """
    Slicer class:
    - Uses ImageLoader object to read images and labels.
    - Creates image slices through the volumes of the images and labels.
    
    Methods:
    - initialize_views(use_standard_axis): Creates a list of unit vectors to use for slicing. 
      If use_standard_axis is True, then use the 3 standard unit vectors instead.
    - sample_slice(): returns a pair of image, label images at a random slice.
    - save_batch_to_folder(batch): saves all images and labels in a given batch as .png images, 
      in folders slicer_batches/images/ and slicer_batches/labels.

    TODO:
    - Generate n random unit vectors in initialize_views()
    - Use views to slice through arbitrary axis in sample_slice().
    """

    def __init__(self, image_loader):
        self.image_loader = image_loader
        self.views = self.initialize_views(use_standard_axis=True)
        self.batch_count = 0

    def initialize_views(self, use_standard_axis=False):
        standard_axis = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

        if use_standard_axis:
            views = standard_axis
        
        return views

    def sample_slice(self):
        view = random.choice(self.views)
        pair = self.image_loader.get_random_pair()
        dims = pair.dims

        # TODO generalize this instead of hardcoded slicing.
        if np.array_equal(view, self.views[0]):
            slice_index = np.random.randint(0, dims[2])
            image_slice = pair.image[:, :, slice_index]
            label_slice = pair.label[:, :, slice_index]
        elif np.array_equal(view, self.views[1]):
            slice_index = np.random.randint(0, dims[1])
            image_slice = pair.image[:, slice_index, :]
            label_slice = pair.label[:, slice_index, :]
        else:
            slice_index = np.random.randint(0, dims[0])
            image_slice = pair.image[slice_index, :, :]
            label_slice = pair.label[slice_index, :, :]

        return (image_slice, label_slice)

    def make_batch(self, batch_size):
        batch = []

        for i in range(batch_size):
            batch.append(self.sample_slice())

        self.batch_count += 1
        return batch

    def save_batch_to_folder(self, batch):
        if not os.path.exists("slicer_batches"):
            print("Creating slicer_batches folder.")
            os.makedirs("slicer_batches")

        batch_path = os.path.join("slicer_batches", "batch" + str(self.batch_count))
        if not os.path.exists(batch_path):
            os.makedirs(batch_path)
            os.makedirs(os.path.join(batch_path, "images"))
            os.makedirs(os.path.join(batch_path, "labels"))
            
        for i in range(len(batch)):
            image, label = batch[i]
            plt.imsave(os.path.join(batch_path,"images", "image" + str(i) + ".png"), image, cmap='Greys_r')
            plt.imsave(os.path.join(batch_path,"labels", "label" + str(i) + ".png"), label, cmap='Greys_r')
        
