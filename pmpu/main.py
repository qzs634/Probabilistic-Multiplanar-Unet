from images.image_loader import ImageLoader
from preprocessing.slicer import Slicer

image_loader = ImageLoader("/home/bib/Documents/DIKU/Bachelor/data_folder/train")
slicer = Slicer(image_loader,
                out_path="/home/bib/Documents/DIKU/Bachelor/Pytorch-UNet/testdata",
                out_image_path="imgs",
                out_label_path="masks")

num_batches = 1
for i in range(num_batches):
    batch = slicer.make_batch(8)
    slicer.save_batch_to_folder(batch)
    print("saved batch no. ", i)

