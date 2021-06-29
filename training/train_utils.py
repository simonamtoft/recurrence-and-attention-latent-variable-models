import os
import wandb
import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
from torchvision.utils import save_image


def lambda_lr(n_epochs, offset, delay):
    """
    Creates learning rate step function for LambdaLR scheduler.
    Stepping starts after "delay" epochs and will reduce LR to 0 when "n_epochs" has been reached
    Offset is used continuing training models.
    """
    if (n_epochs - delay) == 0:
        raise Exception("Error: delay and n_epochs cannot be equal!")
    return lambda epoch: 1 - max(0, epoch + offset - delay)/(n_epochs - delay)


def plt_img_save(img, name='log_image.png'):
    f, ax = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        idx = i*2
        for j in range(2):
            ax[j, i].imshow(np.reshape(img[idx+j, :], (28,28)), cmap='gray')
            ax[j, i].axis('off')
    
    f.savefig(name, transparent=True, bbox_inches='tight')
    plt.close()


def log_images(x_recon, x_sample):
    convert_img(x_recon, "recon")
    convert_img(x_sample, "sample")

    # Log the images to wandb
    wandb.log({
        "Reconstruction": wandb.Image("recon.png"),
        "Sample": wandb.Image("sample.png")
    }, commit=True)

    # Delete the logged images
    os.remove("recon.png")
    os.remove("sample.png")


def convert_img(img, img_name):
    name_jpg = img_name + '.jpg'
    name_png = img_name + '.png'

    # Save batch as single image
    save_image(img, name_jpg)

    # Load image
    imag = image.imread(name_jpg)[:, :, 0]

    # Delete image
    os.remove(name_jpg)

    # Save image as proper plots
    plt_img_save(imag, name=name_png)