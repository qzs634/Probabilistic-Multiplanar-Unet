import os
from images.image_loader  import ImageLoader
from images.image_label_pair  import ImageLabelPair
import random
import uuid
import PIL
import png
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
    - make_batch(n): returns a list of n images and labels.
    - save_batch_to_folder(batch): saves all images and labels in a given batch as .png images, 
      in folders slicer_batches/images/ and slicer_batches/labels.

    TODO:
    - Generate n random unit vectors in initialize_views()
    - Use views to slice through arbitrary axis in sample_slice().
    """

    def __init__(self, image_loader, out_path="slicer_batches", out_image_path="images", out_label_path="labels"):
        self.image_loader = image_loader
        self.views = self.initialize_views(use_standard_axis=True)
        self.batch_count = 0
        self.out_path = out_path
        self.out_image_path = os.path.join(out_path, out_image_path)
        self.out_label_path = os.path.join(out_path, out_label_path)

    def initialize_views(self, use_standard_axis=False):
        standard_axis = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

        if use_standard_axis:
            views = standard_axis
        
        return views

    def sample_slice(self, random_pair=False, index=0):
        view = random.choice(self.views)
        if random_pair:
            pair = self.image_loader.get_random_pair()
        else:
            pair = self.image_loader.get_pair(index)
        dims = pair.dims
        
        # TODO generalize this instead of hardcoded slicing.
        if False: #np.array_equal(view, self.views[0]):
            slice_index = np.random.randint(0, dims[2])
            image_slice = pair.image[:, :, slice_index]
            label_slice = pair.label[:, :, slice_index]
        elif False: #np.array_equal(view, self.views[1]):
            slice_index = np.random.randint(0, dims[1])
            image_slice = pair.image[:, slice_index, :]
            label_slice = pair.label[:, slice_index, :]
        else:
            slice_index = np.random.randint(0, dims[0])
            image_slice = pair.image[slice_index, :, :]
            label_slice = pair.label[slice_index, :, :]

        ret_pair = self.pad_dimensions(image_slice, label_slice)

        return ret_pair

    def make_batch(self, batch_size):
        batch = []

        i = 0
        while (i < batch_size):
            image_pair = self.sample_slice(random_pair=False, index=i%24)
            # Filter applied. Only save images with foreground in labels
            if np.max(image_pair[1]) > 1:
                batch.append(image_pair)
                i += 1

        self.batch_count += 1
        return batch

    def make_all(self):
        pairs = []
        for i in range(self.image_loader.file_count):
            pair = self.image_loader.get_pair(i)
            dims = pair.dims
            # Hardcoded side view
            for slice in range(dims[0]):
                image_slice = pair.image[slice, :, :]
                label_slice = pair.label[slice, :, :]
                if np.max(label_slice) > 1:
                    ret_image, ret_label = self.pad_dimensions(image_slice, label_slice)
                    pairs.append(( ret_image, (ret_label > 1).astype(np.uint8) ))
                    #pairs.append((np.flip(ret_image, axis=1), np.flip(ret_label, axis=1)))

        return pairs


    def save_batch_to_folder(self, batch):
        if not os.path.exists(self.out_path):
            print("Creating {} folder.".format(self.out_path))
            os.makedirs(self.out_path)
            os.makedirs(self.out_image_path)
            os.makedirs(self.out_label_path)
            

        """
        # Save to individual batch folders

        batch_path = os.path.join("slicer_batches", "batch" + str(self.batch_count))
        if not os.path.exists(batch_path):
            os.makedirs(batch_path)
            os.makedirs(os.path.join(batch_path, "images"))
            os.makedirs(os.path.join(batch_path, "labels"))
        """
        
        for i in range(len(batch)):
            file_name = str(uuid.uuid4())
            image, label = batch[i]
            image = (image / np.max(image)) * 255
            """
            if np.max(label) > 0:
                label = (label / np.max(label)) * 255
            """
            # image.astype(np.uint8).copy() hacky fix. Don't know what purpose the .copy() has
            png.from_array(image.astype(np.uint8).copy(), 'L').save(os.path.join(self.out_image_path, file_name + ".png"))
            png.from_array(label.astype(np.uint8).copy(), 'L').save(os.path.join(self.out_label_path, file_name + ".png"))

    def pad_dimensions(self, image, label):
        dim_diff = np.max(image.shape) - np.min(image.shape)
        
        if dim_diff != 0:
            if np.argmin(image.shape) == 0:
                # pad rows
                image = np.vstack((image, np.zeros((dim_diff, image.shape[1]))))
                label = np.vstack((label, np.zeros((dim_diff, label.shape[1]))))
            else:
                # pad columns
                image = np.hstack((image, np.zeros((image.shape[0], dim_diff))))
                label = np.hstack((label, np.zeros((label.shape[0], dim_diff))))

        return (image, label)
                

        
        
