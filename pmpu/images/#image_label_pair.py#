import nibabel as nib
import numpy as np



class ImageLabelPair():
    """
    ImageLabelPair:
    - Stores images and corresponding labels, used for making batches.
    """
    def __init__(self, image, label):
        self.image = image
        self.label = label
        if self.image.shape != self.label.shape:
            print("Mismath in image and label shapes - image shape: {}, label shape: {}".format(
                self.image.shape, self.label.shape))
        self.dims = self.image.shape
