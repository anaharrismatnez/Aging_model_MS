import torch
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import autocast


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def get_figure_AE(
    img: torch.Tensor,
    recons: torch.Tensor,
    display = False,
):
    img_npy_0 = img.squeeze().squeeze().cpu().numpy()
    recons_npy_0 = recons.squeeze().squeeze().detach().cpu().numpy()

    fig, axs = plt.subplots(nrows=1,ncols=2,dpi=300)
    axs[0].imshow(img_npy_0[:,:,83].T, cmap="gray",origin='lower')
    axs[1].imshow(recons_npy_0[:,:,83].T, cmap="gray",origin='lower')
    plt.axis("off")

    if display:
        plt.show()

    return fig


@torch.no_grad()
def get_figure_ldm(
    noise_shape,
    e_c,
    autoencoder,
    diffusion,
    scheduler,
    device,
    scale_factor,
    display = False,
):
    latent = torch.rand(noise_shape).to(device)
    noise = latent
    combined = torch.cat((e_c,noise), dim=1)


    for t in tqdm(scheduler.timesteps, ncols=70):
        noise_pred = diffusion(x=combined, timesteps=torch.asarray((t,)).to(device))
        latent, _ = scheduler.step(noise_pred, t, latent)
        combined = torch.cat((e_c, latent), dim=1)
        if t == 999:
            first = latent

    first_recons = autoencoder.decode_stage_2_outputs(first / scale_factor)
    recons = autoencoder.decode_stage_2_outputs(latent / scale_factor)
    noise_npy_0 = noise.squeeze().squeeze().detach().cpu().numpy()
    noise_pred_npy_0 = noise_pred.squeeze().squeeze().detach().cpu().numpy()
    recons_npy_0 = recons.squeeze().squeeze().detach().cpu().numpy()
    first_recons_npy_0 = first_recons.squeeze().squeeze().detach().cpu().numpy()

    fig, axs = plt.subplots(nrows=2,ncols=2)
    axs[0,0].imshow(noise_npy_0[0,:,:,16].T, cmap="gray",origin='lower')
    axs[0,0].set_title('Diffusion input')
    axs[0,0].axis("off")
    axs[0,1].imshow(noise_pred_npy_0[0,:,:,16].T, cmap="gray",origin='lower')
    axs[0,1].set_title('Diffusion output')
    axs[0,1].axis("off")
    axs[1,0].imshow(first_recons_npy_0[:,:,83].T, cmap="gray",origin='lower')
    axs[1,0].set_title('Reconstruction t=999')
    axs[1,0].axis("off")
    axs[1,1].imshow(recons_npy_0[:,:,83].T, cmap="gray",origin='lower')
    axs[1,1].set_title('Reconstruction t=0')
    axs[1,1].axis("off")

    if display:
        plt.show()

    return fig

@torch.no_grad()
def get_figure_ldm_SR(
    noise_shape,
    LR,
    autoencoder,
    diffusion,
    scheduler,
    device,
    scale_factor,
    display = False,
):
    latent = torch.rand(noise_shape).to(device)
    LR_noise = torch.rand(noise_shape).to(device)
    noise = latent
    noise_level = 10
    noise_level = torch.Tensor((noise_level,)).long().to(device)
    noisy_low_res_image = scheduler.add_noise(original_samples=LR, noise=LR_noise, timesteps=torch.Tensor((noise_level,)).long().to(device))
    scheduler.set_timesteps(num_inference_steps=1000)
    


    for t in tqdm(scheduler.timesteps, ncols=70):
        with autocast(enabled=True):
            combined = torch.cat((latent,noisy_low_res_image), dim=1)
            noise_pred = diffusion(x=combined, timesteps=torch.asarray((t,)).to(device),class_labels=noise_level)
            if t == 999:
                first = latent
        latent, _ = scheduler.step(noise_pred, t, latent)

    first_recons = autoencoder.decode_stage_2_outputs(first / scale_factor)
    recons = autoencoder.decode_stage_2_outputs(latent / scale_factor)
    noise_npy_0 = noise.squeeze().squeeze().detach().cpu().numpy()
    noise_pred_npy_0 = noise_pred.squeeze().squeeze().detach().cpu().numpy()
    recons_npy_0 = recons.squeeze().squeeze().detach().cpu().numpy()
    first_recons_npy_0 = first_recons.squeeze().squeeze().detach().cpu().numpy()

    fig, axs = plt.subplots(nrows=2,ncols=2)
    axs[0,0].imshow(noise_npy_0[0,:,:,16].T, cmap="gray",origin='lower')
    axs[0,0].set_title('Diffusion input')
    axs[0,0].axis("off")
    axs[0,1].imshow(noise_pred_npy_0[0,:,:,16].T, cmap="gray",origin='lower')
    axs[0,1].set_title('Diffusion output')
    axs[0,1].axis("off")
    axs[1,0].imshow(first_recons_npy_0[:,:,83].T, cmap="gray",origin='lower')
    axs[1,0].set_title('Reconstruction t=999')
    axs[1,0].axis("off")
    axs[1,1].imshow(recons_npy_0[:,:,83].T, cmap="gray",origin='lower')
    axs[1,1].set_title('Reconstruction t=0')
    axs[1,1].axis("off")

    if display:
        plt.show()

    return fig



def get_scale_factor(model,image,device):
    with torch.no_grad():
        with autocast(enabled=True):
            z = model.encode_stage_2_inputs(image.to(device))

    scale_factor = 1 / torch.std(z)
    return scale_factor