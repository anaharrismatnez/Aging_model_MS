

import os
import sys
main_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_folder)

import torch
import math
import typing as t
from pathlib import Path
import wandb
from datetime import date
import copy
from utils.utils_image import *
import json
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
import monai


def dim_out_layer(patch_size,p=1,img_size=128,k=4,s=2):
    N = img_size

    if patch_size == 70:
        layers = 5
    elif patch_size == 34:
        layers = 4
    else:
        layers = 3

    for i in range(layers-2):
        out = math.floor(((N - k + 2*p)/s) + 1)
        N = out

    for j in range(2):
        out = math.floor((N - k + 2*p) + 1)
        N = out
    return N


def brain_RMSE_loss(fup,fake):
    rmse_mask = torch.ne(fup, -1)
    rmse_score = torch.sqrt(torch.mean(torch.pow(torch.masked_select(fup, rmse_mask) - torch.masked_select(fake, rmse_mask), 2)))
    return rmse_score


def save_args(path,args):
    """Save input arguments as a json file in args.output_dir"""
    save_json(path+"/args.json", copy.deepcopy(args.__dict__))


def save_json(filename: Path, data: t.Dict):
    """Save dictionary data to filename as a json file"""
    assert type(data) == dict
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            data[key] = value.tolist()
        elif isinstance(value, np.float32):
            data[key] = float(value)
        elif isinstance(value, Path) or isinstance(value, torch.device):
            data[key] = str(value)
    with open(filename, "w") as file:
        json.dump(data, file)


def paths(args,mode='training'):
    today = date.today()
    today = ('').join(str(today).split('-'))

    split_path = args.data_path.split('/')

    #project = split_path[-2]

    if args.n:
        models_path = f'results/{args.n}/training_cGAN/checkpoints'
        root = f'results/{args.n}/training_cGAN'
    else:
        models_path = f'results/run_{today}/training_cGAN/checkpoints'
        root = f'results/run_{today}/training_cGAN'
    
    

    if args.w:
        if args.n:
            wandb.init(project='paper_cGAN',config=args,anonymous="allow", name=f'{today}_{args.n}')
        else:
            wandb.init(project='paper_cGAN',config=args,anonymous="allow", name=f'{today}')

    
    return models_path,root

def move_LR_files(source,target,data_path,basals=False):
    print('moving LR files')
    for file in os.listdir(source):
        if file.startswith('r_'):
            new_name = file.replace('r_','')
            folder = new_name.split('.nii.gz')[0]
            new_name = new_name.replace('.nii.gz','_V0.npy')
            info = json.load(open(os.path.join(data_path,folder,'info.json'),'r'))
            
            if basals:
                if info['delta'] == 0:
                    t,s,m = folder.split('_')
                    basal = f'1_{s}_{m}'
                    print(basal)
                    
                    img = read_nifti(os.path.join(source,file))
                    #img = np.moveaxis(img,0,2)
                    normalized = data_transformation_numpy(img,None,(0,1))

                    np.save(os.path.join(target,new_name),normalized)
                else:
                    continue

            else:
                print(file)
                img = read_nifti(os.path.join(source,file))
                #img = np.moveaxis(img,0,2)
                normalized = data_transformation_numpy(img,None,(0,1))
                np.save(os.path.join(target,new_name),normalized)
            

def move_HR_files(source,data_path):
    print('moving HR files')
    for file in os.listdir(source):
        folder = file.split('_V0.npy')[0]
        print(folder)
        n,s,m = folder.split('_')
        name = s+'_'+m
        delta = json.load(open(os.path.join(data_path,folder,'info.json'),'r'))['delta']
        basal = np.load(os.path.join(data_path,folder,f'{name}.npy'))

        if delta == 0:
            img = basal
        else:
            img = np.load(os.path.join(data_path,folder,f'r_{name}.npy'))

        mask = np.zeros_like(basal)
        mask[basal != 0] = 1
        img[mask == 0] = 0
        normalized = img_normalization(img,range=(0,1))
        np.save(os.path.join(source,f'{n}_{name}_V1.npy'),normalized)


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



def get_scale_factor(model,image,device):
    with torch.no_grad():
        with autocast(enabled=True):
            z = model.encode_stage_2_inputs(image.to(device))

    scale_factor = 1 / torch.std(z)
    return scale_factor


class EarlyStopping: 
    def __init__(self, tolerance=30,min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss):
        if train_loss > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True


def plot_img(basal,img,gradients,index=83,display=None):

    fig, axs = plt.subplots(nrows=1,ncols=3)
    axs[0].imshow(basal[:,:,index].T, cmap='gray',origin='lower')
    axs[0].set_title('Condition')
    axs[0].axis("off")
    axs[1].imshow(img[:,:,index].T, cmap='gray',origin='lower')
    axs[1].set_title('Fake')
    axs[1].axis("off")
    map = axs[2].imshow(gradients[:,:,index].T, cmap='jet',origin='lower')
    axs[2].set_title('Saliency map')
    axs[2].axis("off")
    cbar = fig.colorbar(map,ax=axs[2])

    if display:
        plt.show()
        

    return fig
