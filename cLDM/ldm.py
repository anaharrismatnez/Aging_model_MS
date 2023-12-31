# Code based on: # https://github.com/Warvito/generative_chestxray/blob/main/src/python/training/train_ldm.py

import argparse
import sys
import os
main_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_folder)

import torch.optim as optim
from omegaconf import OmegaConf
from datetime import date
import wandb
import pickle 
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import *
from utils.util import * 

from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from monai.utils import first,set_determinism
from generative.networks.nets import AutoencoderKL


def load_AE(args,device,image):
    config_AE = OmegaConf.load(args.AE_config)
    checkpoint = torch.load(args.AE_model) 
    autoencoder = AutoencoderKL(**config_AE["stage1"]["params"]).to(device)
    autoencoder.load_state_dict(checkpoint['model'])

    scale_factor = get_scale_factor(autoencoder,image,device)

    print('Autoencoder model loaded from:',args.AE_model)

    return autoencoder.to(device),scale_factor.to(device)


def paths(args):
    today = date.today()
    today = ('').join(str(today).split('-'))


    if args.n:
        models_path = f'results/{args.n}/cLDM/checkpoints/'
    else:
        models_path = f'results/run_{today}/cLDM/checkpoints/'
    
    if args.w:
        if args.n:
            wandb.init(project='cLDM',config=args, anonymous="allow", name=f'{today}_{args.n}')
        else:
            wandb.init(project='cLDM',config=args, anonymous="allow", name=f'{today}')

    return models_path

def main(args,device):
    models_path = paths(args)

    # GET DATASET
    if args.data_path.endswith('.pkl'):
        data = pickle.load(open(args.data_path,'rb'))
    else:
        data = get_data(args.data_path)

    train_data = train_dataset(data,'ldm',transform=rest_T)
    train_loader = DataLoader(train_data,batch_size=args.B,num_workers=0,shuffle=True)
    
    # DIFFUSION MODEL CONFIGURATION
    config = OmegaConf.load(args.config)
    diffusion = DiffusionModelUNet(**config["ldm"]["params"]).to(device)
    scheduler = DDPMScheduler(**config["ldm"].get("scheduler", dict()))
    optimizer = optim.Adam(diffusion.parameters(), lr=config["ldm"]["base_lr"])
    autoencoder,scale_factor = load_AE(args,device,first(train_loader)[0])

    autoencoder.eval()

    best_loss = float("inf")
    start_epoch = 0
    counter = 0

    if not os.path.exists(models_path):
        os.makedirs(models_path)
    else:
        checkpoint = torch.load(f'{models_path}/{args.e0}.pth') 
        diffusion.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint['best_loss']
        print('Model loaded from checkpoint at epoch:',start_epoch)
 

    # TRAINING
    diffusion.train()
    scaler = GradScaler()

    for ep in range(start_epoch,args.e):
        run_loss = 0
        print(f'epoch: {ep+1}/{args.e}')
        for i,data in tqdm(enumerate(train_loader),total=len(train_loader)):

            fup,basal,delta = data
            fup = fup.to(device).float()
            basal = basal.to(device).float()
            delta = delta.unsqueeze(1).to(device)

            timesteps = torch.randint(0, config['ldm']['scheduler']['num_train_timesteps'], (fup.shape[0],), device=device).long()

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                with torch.no_grad():
                    e = autoencoder.encode_stage_2_inputs(fup) * scale_factor

                e_b = autoencoder.encode_stage_2_inputs(basal) * scale_factor
                delta = delta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
                delta = delta.repeat(1, 1, e_b.shape[2], e_b.shape[3], e_b.shape[4])

                e_c = torch.cat([e_b,delta], dim=1) 

                noise = torch.randn_like(e).to(device)
                noisy_e =  scheduler.add_noise(original_samples=e,noise=noise,timesteps=timesteps)

                combined = torch.cat((e_c,noisy_e), dim=1)
                
                noise_pred = diffusion(x=combined,timesteps=timesteps)

                # Use v-prediction parameterization
                if config['ldm']['scheduler']['prediction_type'] == 'v_prediction':
                    target = scheduler.get_velocity(noisy_e, noise, timesteps) 
                else:
                    target = noise
            
                loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            run_loss += loss
        if args.w:
            wandb.log({
                "epoch": ep +1,
                "G_lr": get_lr(optimizer),
                "total_loss": (run_loss.item())/len(train_loader),
            })

            if (ep + 1) % 10 == 0:
                fig = get_figure_ldm(e.shape,e_c,autoencoder,diffusion,scheduler,device,scale_factor)
                plots = wandb.Image(fig)
                plt.close(fig)
                wandb.log({f"epoch {(ep+1)}": plots}) 


        # SAVE BEST MODEL
        if (run_loss.item())/len(train_loader) <= best_loss:
                best_loss = (run_loss.item())/len(train_loader)
                best_epoch = ep + 1
                checkpoint = {
                    "epoch": ep + 1,
                    "model": diffusion.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_loss": best_loss,
                }
                torch.save(checkpoint, f'{models_path}/best_model.pth')
                if args.w:
                    wandb.save(f'{models_path}/best_model.pth') 
                print('Checkpoint saved!')
                counter = 0
        else:
            counter += 1


        if (ep + 1) % 50 == 0:
            checkpoint = {
                "epoch": ep + 1,
                "model": diffusion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": (run_loss.item())/len(train_loader),
                "best_loss": best_loss,
            }
            torch.save(checkpoint, f'{models_path}/{ep+1}.pth') 
            print('50 epochs: Checkpoint saved!')
    
    # SAVE LAST MODEL

    print('Training completed')  
    checkpoint = {
                "epoch": ep + 1,
                "model": diffusion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss":(run_loss.item())/len(train_loader),
                "best_loss": best_loss,
            }
    torch.save(checkpoint, f'{models_path}/final_model.pth')
    if args.w:
        wandb.save(f'{models_path}/final_model.pth') 



if __name__ == "__main__":

    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path',required=True, type=str,help='Data path')
    parser.add_argument('-n',required=False, type=str,help='Experiment name')
    parser.add_argument('-B',default=1, type=int,  help='Batch size')
    parser.add_argument('-e',default=25, type=int, help='Number of epochs')
    parser.add_argument('-e0',required=False, type=str, help='Epoch to load model')
    parser.add_argument('-w', required=False, type=bool, help= 'True if wandb initilization is required')

    parser.add_argument('-config',required=True, type=str, help= 'Config file')
    parser.add_argument('-AE_model',required=True, type=str, help= 'Path to autoencoder model.')
    parser.add_argument('-AE_config',required=True, type=str, help= 'Path to autoencoder model.')

    args = parser.parse_args()

    set_determinism(seed=2)

    main(args,device)

