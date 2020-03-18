import nibabel as nib
import numpy as np



class ImageLabelPair():

    def __init__(self, image, label):
        self.image = image
        self.label = label
        if self.image.shape == self.label.shape:
            print("Mismath in image and label shapes - image shape: {}, label shape: {}".format(
                self.image.shape, self, label.shape
            ))
        self.dims = self.image.shape
