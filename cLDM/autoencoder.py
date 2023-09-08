
# Code based on: # https://github.com/Warvito/generative_chestxray/blob/main/src/python/training/train_aekl.py

import argparse
import os
from omegaconf import OmegaConf
import torch
import torch.optim as optim
import sys
main_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_folder)

from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL
from generative.networks.nets.patchgan_discriminator import PatchDiscriminator
from generative.losses.adversarial_loss import PatchAdversarialLoss
from datetime import date
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from dataset import *
from utils.util import * 
import pickle


torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def paths(args):
    today = date.today()
    today = ('').join(str(today).split('-'))


    if args.n:
        models_path = f'results/{args.n}/autoencoder/checkpoints/'
    else:
        models_path = f'results/run_{today}/autoencoder/checkpoints/'
    
    if args.w:
        if args.n:
            wandb.init(project='autoencoder',config=args, anonymous="allow", name=f'{today}_{args.n}')
        else:
            wandb.init(project='autoencoder',config=args, anonymous="allow", name=f'{today}')

    return models_path


def main(args):
    models_path = paths(args)

    # MODEL CONFIGURATION
    config = OmegaConf.load(args.config)
    model = AutoencoderKL(**config["stage1"]["params"]).to(device)
    D = PatchDiscriminator(**config["discriminator"]["params"]).to(device)
    perceptual_loss = PerceptualLoss(**config["perceptual_network"]["params"]).to(device)

    G_optimizer = optim.Adam(model.parameters(), lr=config["stage1"]["base_lr"])
    D_optimizer = optim.Adam(D.parameters(), lr=config["stage1"]["disc_lr"])

    best_loss = float("inf")
    start_epoch = 0
    counter = 0

    if not os.path.exists(models_path):
        os.makedirs(models_path)
    else:
        checkpoint = torch.load(f'{models_path}/best_model.pth') 
        model.load_state_dict(checkpoint['model'])
        D.load_state_dict(checkpoint["discriminator"])
        G_optimizer.load_state_dict(checkpoint["G_optimizer"])
        D_optimizer.load_state_dict(checkpoint["D_optimizer"]) 
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
        print('Model loaded from checkpoint at epoch:',start_epoch)    

    # GET DATASET
    if args.data_path.endswith('.pkl'):
        data = pickle.load(open(args.data_path,'rb'))
    else:
        data = get_data_AE(args.data_path)

    if args.agm:
        train_data = train_dataset(data,'autoencoder',augmentation_T)
    else:
        train_data = train_dataset(data,'autoencoder')

    train_loader = DataLoader(train_data,batch_size=args.B,num_workers=0,shuffle=True)

    model.train()
    D.train()

    adv_loss = PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=False)

    # TRAINING

    G_scaler = GradScaler()
    D_scaler = GradScaler()

    for ep in range(start_epoch,args.e):

        run_G_total_loss = run_L1_loss = run_p_loss = run_kl_loss = run_G_loss = run_D_loss = 0

        print(f'epoch: {ep+1}/{args.e}')
        for i,fup in tqdm(enumerate(train_loader),total=len(train_loader)):
            fup = fup.to(device).float()

            # GENERATOR
            G_optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                reconstruction, z_mu, z_sigma = model(x=fup)

                if config["stage1"]["adv_weight"]:
                    logits_fake = D(reconstruction.contiguous().float())[-1]
                    G_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)

                else:
                    G_loss = torch.tensor([0.0]).to(device)

                L1_loss = torch.nn.functional.l1_loss(reconstruction.float(), fup.float())
                p_loss = perceptual_loss(reconstruction.float(), fup.float())

                kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

                G_total_loss = L1_loss + (config["stage1"]["kl_weight"] * kl_loss) + (config["stage1"]["perceptual_weight"] * p_loss) + (config["stage1"]["adv_weight"] * G_loss)

            run_G_total_loss += G_total_loss
            run_L1_loss += L1_loss
            run_p_loss += p_loss
            run_kl_loss += kl_loss
            run_G_loss += G_loss

            G_scaler.scale(G_total_loss).backward()
            G_scaler.step(G_optimizer)
            G_scaler.update()

            # DISCRIMINATOR

            if config["stage1"]["adv_weight"] > 0:
                D_optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=True):
                    #with torch.no_grad():
                    logits_fake = D(reconstruction.contiguous().detach())[-1]
                    D_fake_loss = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                    logits_real = D(fup.contiguous().detach())[-1]
                    D_real_loss = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                    D_loss = (D_fake_loss + D_real_loss) * 0.5
                    
                    D_total_loss = (config["stage1"]["adv_weight"] * D_loss)

                D_scaler.scale(D_total_loss).backward()
                D_scaler.step(D_optimizer)
                D_scaler.update()

            else:
                D_loss = torch.tensor([0.0]).to(device)


            run_D_loss += D_loss


        if args.w:
            wandb.log({
                "epoch": ep +1,
                "G_lr": get_lr(G_optimizer),
                "D_lr": get_lr(D_optimizer),
                "G_total_loss": (run_G_total_loss.item())/len(train_loader),
                "L1_loss": (run_L1_loss.item())/len(train_loader),
                "kl_loss": (run_kl_loss.item())/len(train_loader),
                "G_loss": (run_G_loss.item())/len(train_loader),
                "P_loss": (run_G_loss.item())/len(train_loader),
                "D_loss": (run_D_loss.item())/len(train_loader)
            })

            fig = get_figure_AE(fup,reconstruction)
            plots = wandb.Image(fig)
            plt.close(fig)
            wandb.log({f"epoch {(ep+1)}": plots}) 


        if (run_G_total_loss.item())/len(train_loader) <= best_loss:
                best_loss = (run_G_total_loss.item())/len(train_loader)
                best_epoch = ep +1
                checkpoint = {
                    "epoch": ep + 1,
                    "model": model.state_dict(),
                    "discriminator": D.state_dict(),
                    "G_optimizer": G_optimizer.state_dict(),
                    "D_optimizer": D_optimizer.state_dict(),
                    "best_loss": best_loss,
                }
                torch.save(checkpoint, f'{models_path}/best_model.pth') 

                if args.w:
                    wandb.save(f'{models_path}/best_model.pth')

                print('Checkpoint saved!')
                counter = 0
        else:
            counter += 1

        if counter >= 10:
            print('The model doesnt improve')
            break

    print('Training completed')  
    checkpoint = {
                "epoch": ep + 1,
                "model": model.state_dict(),
                "discriminator": D.state_dict(),
                "G_optimizer": G_optimizer.state_dict(),
                "D_optimizer": D_optimizer.state_dict(),
                "loss":(L1_loss.item())/len(train_loader),
            }
    torch.save(checkpoint, f'{models_path}/final_model.pth')
    if args.w:
        wandb.save(f'{models_path}/final_model.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path',required=True, type=str,help='Data path')
    parser.add_argument('-n',required=False, type=str,help='Experiment name')
    parser.add_argument('-B',default=1, type=int,  help='Batch size')
    parser.add_argument('-e',default=50, type=int, help='Number of epochs')
    parser.add_argument('-w', required=False, type=bool, help= 'True if wandb initilization is required')
    parser.add_argument('-config',required=True, type=str, help= 'Config file')
    parser.add_argument('-agm',default=True, type=bool, help= 'If data augmentation is required')

    args = parser.parse_args()

    main(args)

