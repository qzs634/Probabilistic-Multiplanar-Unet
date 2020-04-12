import sys
import os
from images.image_loader import ImageLoader
from preprocessing.slicer import Slicer

path = sys.argv[1] if len(sys.argv) > 1 else "."
image_loader = ImageLoader(os.path.join(path, "train"))
slicer = Slicer(image_loader,
                out_path="data",
                out_image_path="imgs",
                out_label_path="masks")


num_batches = 1
for i in range(num_batches):
    batch = slicer.make_batch(10)
    slicer.save_batch_to_folder(batch)
    print("saved batch no. ", i)



#slicer.make_all_pairs()