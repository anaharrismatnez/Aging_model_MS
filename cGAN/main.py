# Ana Harris 23/02/2023
# Main file to perform cGAN training
# Model based on pix2pix paper: https://arxiv.org/pdf/1611.07004v1.pdf

import os
import sys
main_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_folder)

from model_pix2pix import *
from train import *
from utils.util import *
from dataset import *
import torch
import wandb
from utils.utils_image import *
from tqdm import tqdm
import argparse
from time import process_time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args,epochs_check):
    t_start = process_time()

    if device == 'cpu':
        raise ValueError ('CUDA not available')

    data = get_data(args.d)
    G = FiLMed_Generator().to(device)
    train_data = dataset(data,mode='training')

    D = Discriminator(p=args.pad,patch_size=args.p).to(device)


    G_optimizer = torch.optim.Adam(G.parameters(),lr=args.lr,betas=(0.5,0.999))
    D_optimizer = torch.optim.Adam(D.parameters(),lr=args.lr,betas=(0.5,0.999))


    models_path,root = paths(args,mode='training')
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    else:
        G_checkpoint = torch.load(models_path+'/best_model_generator_rmse')
        G.load_state_dict(G_checkpoint['model_state_dict']) 
        G_optimizer.load_state_dict(G_checkpoint['optimizer_state_dict'])
        epochs_check = G_checkpoint['epoch']

        D_checkpoint = torch.load(models_path+'/best_model_discriminator_rmse')
        D.load_state_dict(D_checkpoint['model_state_dict'])
        D_optimizer.load_state_dict(D_checkpoint['optimizer_state_dict'])

        print('Model loaded from checkpoint at epoch:',G_checkpoint['epoch'])

    save_args(root,args)

    train(args,train_data,epochs_check,G_optimizer,D_optimizer,device,G,D,models_path)

    if args.w:
        wandb.finish()

    t_finish = process_time()
    print(f'Done. The whole process lasted: {(t_finish-t_start)/60} mins')



if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',required=True, type=str,help='input folder')
    parser.add_argument('-n',required=False, type=str,help='Experiment name')
    parser.add_argument('-B',default=8, type=int,  help='Batch size')
    parser.add_argument('-e',default=100, type=int, help='Number of epochs')
    parser.add_argument('-w', required=False, type=bool, help= 'True if wandb initilization is required')
    parser.add_argument('-l',default=100, type=int, help='Lambda, L1 regularization')
    parser.add_argument('-p',default=34, type=int, help='Patch size of the discriminator')
    parser.add_argument('-s',required=False,default=None, type=bool, help='If smooth_label is required')
    parser.add_argument('-pad',required=False,default=1, type=int, help='Padding at Discriminator')
    parser.add_argument('-g',default=200, type=int, help='Gamma, rmse loss')
    parser.add_argument('-mu_fm_loss',default=20, type=int, help='Mu,feature matching loss')
    parser.add_argument('-lr',default=2e-4, type=int, help='learning_rate')
    parser.add_argument('-vd',default=0.001, type=int, help='upsilon discriminator, L2 regularization')
    parser.add_argument('-vg',default=0.001, type=int, help='upsilon generator, L2 regularization')
    
    args = parser.parse_args()

    epochs_check = 0               

    main(args,epochs_check)

    
    