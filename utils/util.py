

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

    split_path = args.d.split('/')

    project = split_path[-2]

    if args.n:
        models_path = f'results/{args.n}/training_cGAN/checkpoints'
        root = f'results/{args.n}/training_cGAN'
    else:
        models_path = f'results/run_{today}/training_cGAN/checkpoints'
        root = f'results/run_{today}/training_cGAN'

    if args.w:
        if args.n:
            wandb.init(project=project,config=args,anonymous="allow", name=f'{today}_{args.n}')
        else:
            wandb.init(project=project,config=args,anonymous="allow", name=f'{today}')



    return models_path,root


def move_LR_files(source,target,data_path):
    for file in os.listdir(source):
            print(file)
            folder = file.split('.npy')[0]
            info = json.load(open(os.path.join(data_path,folder,'info.json'),'r'))
            shape = info['shape']
            img = np.load(os.path.join(source,file))
            img = np.moveaxis(img,0,2)
            normalized = data_transformation_numpy(img,tuple(shape))
            
            new_name = file.replace('.npy','_V0.npy')
            np.save(os.path.join(target,new_name),normalized)
            

def move_HR_files(source,target,basals=False):
    for folder in os.listdir(source):
        print(folder)
        delta = json.load(open(os.path.join(source,folder,'info.json'),'r'))['delta']
        if basals:
            if delta == 0:
                name = folder.split('1_')[-1]
                img = np.load(os.path.join(source,folder,f'{name}.npy'))
                normalized = img_normalization(img,range=(0,1))
                np.save(os.path.join(target,f'1_{name}_V1.npy'),normalized)

        else:
            if delta != 0:
                n,s,m = folder.split('_')
                name = s+'_'+m
                img = np.load(os.path.join(source,folder,f'r_{name}.npy'))
                basal = np.load(os.path.join(source,folder,f'{name}.npy'))
                mask = np.zeros_like(basal)
                mask[basal != 0] = 1
                img[mask == 0] = 0
                normalized = img_normalization(img,range=(0,1))
                np.save(os.path.join(target,f'{n}_{name}_V1.npy'),normalized)