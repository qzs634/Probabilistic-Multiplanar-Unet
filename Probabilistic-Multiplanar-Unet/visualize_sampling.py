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
    for z_0 in range(-(n_preds//2), (n_preds//2) + 1):
        zs = []
        for z_1 in range(-(n_preds//2), (n_preds//2) + 1):
            z = torch.Tensor([1*z_0*sigma[0] + mu[0], 1*z_1*sigma[1] + mu[1], mu[2], mu[3], mu[4], mu[5]])
            with torch.no_grad():
                sample = train.predict(slice, true_mask, z=z)
            mask = train.mask_to_image(sample, prediction=True)
            zs.append(mask)
        predictions.append(zs)

    plt.rcParams["figure.figsize"] = (10, 10)
    plt.axis('off')
    plt.imsave("viz_scan.png", slice.cpu().numpy().squeeze(), cmap="Greys_r")
    mask_img = train.mask_to_image(true_mask, prediction=False).cpu().numpy().squeeze().transpose(1, 2, 0)
    plt.imsave("viz_label.png", mask_img.astype(np.uint8) * 255)

    fig, ax = plt.subplots(n_preds, n_preds, constrained_layout=True)

    i = 0
    for z_0 in range(n_preds):
        for z_1 in range(n_preds):
            pred = predictions[z_0][z_1]
            pred = pred.cpu().numpy().squeeze().transpose(1, 2, 0)
            ax[z_0, z_1].imshow(pred.astype(np.uint8) * 255)

    plt.setp(ax, xticks=[], yticks=[])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    #plt.axis('off')
    fig.savefig("viz_grid.png", dpi=600, cmap="Greys_r")
    plt.show()


if __name__ == "__main__":
    trainer = ProbUNetTrainer(device, n_channels=1, n_classes=3,
                              load_model=r"C:\Users\Niklas Magnussen\Desktop\TheBachelor\Pytorch-UNet-modified\checkpoints\probabilisticunet.pt", latent_dim=6)
    #trainer.net.eval()

    dir_img = r"C:\Users\Niklas Magnussen\Desktop\TheBachelor\data_folder(Multiclass)\train\images"  # "data/imgs/"
    dir_mask = r"C:\Users\Niklas Magnussen\Desktop\TheBachelor\data_folder(Multiclass)\train\labels"  # "data/masks/"
    dataset = MRI_Dataset(dir_img, dir_mask, trainer.net.n_classes)

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)

    pair = next(iter(loader))
    img = pair["image"].to(device=device, dtype=torch.float32)
    mask = pair["mask"].to(device=device, dtype=torch.float32)
    with torch.no_grad():
        trainer.net.forward(img, mask)

    mu = trainer.net.prior_latent_space.base_dist.loc
    print("mu: ", mu)
    mu = mu.squeeze()
    sigma = trainer.net.prior_latent_space.base_dist.scale
    print("sigma: ", sigma)
    sigma = sigma.squeeze()*40.0

    visualize_sample(trainer, img, mask, 3, mu.squeeze(), sigma.squeeze())


