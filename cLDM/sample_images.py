import argparse
from pathlib import Path
from torch.utils.data import DataLoader

import numpy as np
import torch
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from utils.dataset import *
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
import pickle

# https://github.com/Warvito/generative_chestxray/blob/main/src/python/testing/sample_images.py


def main(args):

    #model_dir = args.model_dir+'/checkpoints/best_model.pth'
    output_dir = Path(args.output_dir+'/sampled_images')
    output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda")

    config = OmegaConf.load(args.config_AE)
    stage1 = AutoencoderKL(**config["stage1"]["params"])
    A_checkpoint = torch.load(args.AE)
    stage1.load_state_dict(A_checkpoint['model'])
    stage1.to(device)
    stage1.eval()

    config = OmegaConf.load(args.config)
    diffusion = DiffusionModelUNet(**config["ldm"].get("params", dict()))
    d_checkpoint = torch.load(args.model_dir)
    #diffusion.load_state_dict(d_checkpoint)
    diffusion.load_state_dict(d_checkpoint['model'])
    diffusion.to(device)
    diffusion.eval()

    scheduler = DDPMScheduler(
        num_train_timesteps=config["ldm"]["scheduler"]["num_train_timesteps"],
        beta_start=config["ldm"]["scheduler"]["beta_start"],
        beta_end=config["ldm"]["scheduler"]["beta_end"],
        schedule=config["ldm"]["scheduler"]["schedule"],
        prediction_type=config["ldm"]["scheduler"]["prediction_type"],
        clip_sample=False,
    )
    scheduler.set_timesteps(args.num_inference_steps)

    # GET DATASET
    if args.d.endswith('.pkl'):
        data = pickle.load(open(args.d,'rb'))
    else:
        data = get_data(args.d)

    test_data = train_dataset(data,'ldm')
    test_loader = DataLoader(test_data)

    ensemble = []

    for i,data in enumerate(test_loader):
        noise = torch.randn((1, config["ldm"]["params"]["spatial_dims"], args.x_size, args.y_size,args.z_size)).to(device)
        current_img = noise
        fup,basal,delta = data # MODIFICAR TEST SET
        basal = basal.unsqueeze(1).to(device).float()

        with autocast(enabled=True):

            z_mu_b,z_sigma_b = stage1.encode(basal)
            e_b = stage1.sampling(z_mu_b,z_sigma_b)

            # EXPAND DELTA DIMENSIONS 2 CONCATENATE BASAL (CONDITIONS)
            delta = delta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
            delta = delta.repeat(1, 1, e_b.shape[2], e_b.shape[3], e_b.shape[4]).to(device)
            e_c = torch.cat([e_b,delta], dim=1)

            combined = torch.cat((e_c,noise), dim=1)

            progress_bar = tqdm(scheduler.timesteps)
            for t in progress_bar:
                with torch.no_grad():
                    model_output = diffusion(combined, timesteps=torch.asarray((t,)).to(current_img.device))
                    current_img, _ = scheduler.step(model_output, t, current_img)  # this is the prediction x_t at the time step t
                    combined = torch.cat((e_c, current_img), dim=1)


            sample = stage1.decode(current_img)

            sample = sample.squeeze().squeeze().detach().cpu().numpy() 
            ensemble.append(sample)
        plt.figure(dpi=300)
        plt.imshow(sample[83,:,:].T, cmap="gray",origin='lower')
        plt.axis("off")
        plt.savefig(f'{output_dir}/sample_{i}.png')
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d',default='/home/extop/Ana/Datasets/Niftis_sans_Workspaces_biogen/numpys.pkl', type=str,help='Data path') # PATIENTS
    parser.add_argument("--model_dir", help="Path to the .pth model from the diffusion model.")
    parser.add_argument("--AE", default='results/test9_agm/autoencoder/checkpoints/best_model.pth',help="Path to the .pth model from the stage1.")
    parser.add_argument("--config",default='configs/ldm.yaml', type=str, help= 'Config file')
    parser.add_argument("--config_AE",default='configs/autoencoder_test9.yaml', type=str, help= 'Config file')
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="")
    parser.add_argument("--x_size", type=int, default=32, help="Latent space x size.")
    parser.add_argument("--y_size", type=int, default=32, help="Latent space y size.")
    parser.add_argument("--z_size", type=int, default=32, help="Latent space z size.")
    parser.add_argument("--scale_factor",default=1.0, type=float, help="Latent space y size.")
    parser.add_argument("--num_inference_steps",default=1000, type=int, help="")
    parser.add_argument("--output_dir" ,type=str, help="")

    args = parser.parse_args()

    main(args)