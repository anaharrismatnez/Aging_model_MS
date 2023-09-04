# Code based on: https://github.com/Warvito/generative_chestxray/blob/main/src/python/testing/sample_images.py

import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import sys
import os
main_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_folder)

import torch
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from omegaconf import OmegaConf
from tqdm import tqdm
from dataset import *
from utils.util import * 
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
import pickle
from monai.utils import first,set_determinism


def load_AE(args,device,image):
    config_AE = OmegaConf.load(args.AE_config)
    checkpoint = torch.load(args.AE) 
    autoencoder = AutoencoderKL(**config_AE["stage1"]["params"]).to(device)
    autoencoder.load_state_dict(checkpoint['model'])

    scale_factor = get_scale_factor(autoencoder,image,device)

    print('Autoencoder model loaded from:',args.AE)
    print('Scale factor:',scale_factor)

    return autoencoder.to(device),scale_factor.to(device)




def main(args):

    output_dir = Path(args.output_dir+'/frames/')
    output_dir.mkdir(exist_ok=True, parents=True)

    output_dir_npy = Path(args.output_dir+'/LR')
    output_dir_npy.mkdir(exist_ok=True, parents=True)

    # GET DATASET
    if args.d.endswith('.pkl'):
        data = pickle.load(open(args.d,'rb'))
    else:
        data = get_data(args.d)

    test_data = test_dataset(data,rest_T)
    test_loader = DataLoader(test_data)

    device = torch.device("cuda")

    stage1,scale_factor = load_AE(args,device,first(test_loader)[0])
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


    for i,data in enumerate(test_loader):
        noise = torch.randn((1, config["ldm"]["params"]["spatial_dims"], args.x_size, args.y_size,args.z_size)).to(device)
        current_img = noise
        basal,delta,filename,shape = data 

        basal = basal.to(device)

        with autocast(enabled=True):

            e_b = stage1.encode_stage_2_inputs(basal) #* scale_factor

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


            sample = stage1.decode_stage_2_outputs(current_img/scale_factor)

            sample = sample.squeeze().squeeze().detach().cpu().numpy() 

        filename = str(filename).split("'")[1]

        size = (int(shape[0][0]),int(shape[1][0]),int(shape[2][0]))
        transform = monai.transforms.Compose([
            monai.transforms.AddChannel(),
            monai.transforms.ResizeWithPadOrCrop(spatial_size=size),
            monai.transforms.ScaleIntensity(minv=0.0, maxv=1.0)
            ])
        sample = transform(sample)

        sample = sample.squeeze().numpy() 

        np.save(f'{output_dir_npy}/{filename}.npy',sample) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path',required=True, type=str,help='Data path') # PATIENTS
    parser.add_argument("-model_dir",required=True, help="Path to the .pth model from the diffusion model.")
    parser.add_argument("-AE_model", required=True,help="Path to the .pth model from the stage1.")
    parser.add_argument("-config",required=True, type=str, help= 'Config file')
    parser.add_argument("-AE_config",required=True, type=str, help= 'Config file')
    parser.add_argument("-guidance_scale", type=float, default=7.0, help="")
    parser.add_argument("-x_size", type=int, default=32, help="Latent space x size.")
    parser.add_argument("-y_size", type=int, default=32, help="Latent space y size.")
    parser.add_argument("-z_size", type=int, default=32, help="Latent space z size.")
    parser.add_argument("-num_inference_steps",default=1000, type=int, help="")
    parser.add_argument("-output_dir" ,required=True,type=str, help="Output directory")

    args = parser.parse_args()

    set_determinism(seed=2)

    main(args)