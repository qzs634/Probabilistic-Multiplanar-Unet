import os
import random
import nibabel as nib
from images.image_label_pair import ImageLabelPair


class ImageLoader():
    """
    ImageLoader class:
    - Reads 3d volume and corresponding label from file.
    - Images and labels are read from the images/ and labels/ subfolder in path.

    Methods:
    - get_pair(index): returns an ImageLabelPair of the image and label at index.
    - get_random_pair(): returns ImageLabelPair at random index.
    - get_n_random_pairs(n): returns list of n ImageLabelPair objects at random indices.

    """  

    def __init__(self, path):
        self.path = path
        self.image_path = os.path.join(self.path, "images")
        self.label_path = os.path.join(self.path, "labels")
        self.image_index = 0
        self.file_count = len(os.listdir(self.image_path))

    def get_pair(self, index=None):
        if index == None:
            index = self.image_index

            
        try:
            image_at_path = os.listdir(self.image_path)[index]
            image = nib.load(os.path.join(self.image_path, image_at_path))
            label = nib.load(os.path.join(self.label_path, image_at_path))
            index += 1
            return ImageLabelPair(image.get_fdata(), label.get_fdata())
        except FileNotFoundError:
            print("image{}.nii not found in {} subfolders.".format(index, self.path))
            return None

    def get_random_pair(self):
        i = random.randint(0, self.file_count - 1)
        return self.get_pair(index=i)

    def get_n_random_pairs(self, n):
        pairs = []
        
        try:    
            indices = random.sample(range(0, self.file_count), n)
        except ValueError:
            print("n must not exceed ImageLoader.file_count.")

        for i in indices:
            pairs.append(self.get_pair(index=i))

        return pairs
        
      
      










