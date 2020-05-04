import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, random_split
from trainer import ProbUNetTrainer
from utils.mri_dataset import MRI_Dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def visualize_sample(train, slice, true_mask, n_preds, mu, sigma):
    """
    load a model
    load a slice
    use the model to sample n predictions from the slice
    plt.show
    
    create a n_preds x n_preds grid of predictions 
    """

    predictions = []
    i = 0
    for z_0 in range(n_preds):
        zs = []
        for z_1 in range(n_preds):
            z = torch.Tensor([(z_0 - mu[0])/sigma[0], (z_1 - mu[1])/sigma[1]])
            sample = train.predict(slice, true_mask)
            mask = train.mask_to_image(sample, prediction=True)
            zs.append(mask)
        predictions.append(zs)


    fig, ax = plt.subplots(1 + n_preds, n_preds)
    ax[0][0].imshow(slice.cpu().numpy().squeeze(), cmap="Greys_r")
    mask_img = train.mask_to_image(true_mask, prediction=False).cpu().numpy().squeeze().transpose(1, 2, 0)
    ax[0][1].imshow(mask_img.astype(np.uint8) * 255)
    i = 0
    for z_0 in range(n_preds):
        for z_1 in range(n_preds):
            pred = predictions[z_0][z_1]
            pred = pred.cpu().numpy().squeeze().transpose(1, 2, 0)
            ax[z_0 + 1, z_1].imshow(pred.astype(np.uint8) * 255)
    plt.show()


if __name__ == "__main__":
    trainer = ProbUNetTrainer(device, n_channels=1, n_classes=4,
                              load_model=r"C:\Users\Niklas Magnussen\Desktop\TheBachelor\Pytorch-UNet-modified\checkpoints\model.pth", latent_dim=2)
    #trainer.net.eval()

    dir_img = r"C:\Users\Niklas Magnussen\Desktop\TheBachelor\data_folder\train\images"  # "data/imgs/"
    dir_mask = r"C:\Users\Niklas Magnussen\Desktop\TheBachelor\data_folder\train\labels"  # "data/masks/"
    dataset = MRI_Dataset(dir_img, dir_mask, trainer.net.n_classes)

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)

    pair = next(iter(loader))
    img = pair["image"].to(device=device, dtype=torch.float32)
    mask = pair["mask"].to(device=device, dtype=torch.float32)
    with torch.no_grad():
        trainer.net.forward(img, mask)

    mu = trainer.net.prior_latent_space.base_dist.loc
    mu = mu.squeeze()
    sigma = trainer.net.prior_latent_space.base_dist.scale
    sigma = sigma.squeeze()

    visualize_sample(trainer, img, mask, 2, mu.squeeze(), sigma.squeeze())


