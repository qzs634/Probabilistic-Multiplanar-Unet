from images.image_loader import ImageLoader
from preprocessing.slicer import Slicer

image_loader = ImageLoader("/home/bib/Documents/DIKU/Bachelor/data_folder/train")
slicer = Slicer(image_loader)

for i in range(3):
    batch = slicer.make_batch(5)
    slicer.save_batch_to_folder(batch)
    print("saved batch no. ", i)
