# Ana Harris 23/02/2023
# Main file to perform cGAN training
# Model based on pix2pix paper: https://arxiv.org/pdf/1611.07004v1.pdf

import os
import sys
main_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_folder)

from train import *
from utils.util import *
from dataset import *
import torch
from utils.utils_image import *
import argparse
from time import process_time
from model_pix2pix import Generator,Discriminator

import pickle
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args,epochs_check):
    t_start = process_time()

    if device == 'cpu':
        raise ValueError ('CUDA not available')

    # TRAINING DATA:
    print('Training data:')
    data = get_data(args.data_path)
    #data = pickle.load(open(args.data_path,'rb'))
    train_data = dataset(data,mode='training')
    train_loader = DataLoader(train_data, batch_size=args.B, num_workers=0, shuffle=True,drop_last=True)

    # VALIDATION DATA:

    if args.val_path:
        print('Validation data:')
        data = get_data(args.val_path)
        #data = pickle.load(open(args.val_path,'rb'))
        val_data = dataset(data,mode='validation') 
        val_loader = DataLoader(val_data, batch_size=args.B, num_workers=0, shuffle=True)
    else:
        val_loader = None

        
    G = Generator().to(device)
    D = Discriminator(p=args.pad,patch_size=args.p).to(device)

    G_optimizer = torch.optim.Adam(G.parameters(),lr=args.lr,betas=(0.5,0.999))
    D_optimizer = torch.optim.Adam(D.parameters(),lr=args.lr,betas=(0.5,0.999))


    models_path,root = paths(args,mode='training')
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    else:
        G_checkpoint = torch.load(models_path+'/best_model_generator')
        G.load_state_dict(G_checkpoint['model_state_dict']) 
        G_optimizer.load_state_dict(G_checkpoint['optimizer_state_dict'])
        epochs_check = G_checkpoint['epoch']

        D_checkpoint = torch.load(models_path+'/best_model_discriminator')
        D.load_state_dict(D_checkpoint['model_state_dict'])
        D_optimizer.load_state_dict(D_checkpoint['optimizer_state_dict'])

        print('Model loaded from checkpoint at epoch:',G_checkpoint['epoch']) 

    save_args(root,args)

    train(args,train_loader,val_loader,epochs_check,G_optimizer,D_optimizer,device,G,D,models_path)

    t_finish = process_time()
    print(f'Done. The whole process lasted: {(t_finish-t_start)/60} mins')



if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path',required=True, type=str,help='Training set path')
    parser.add_argument('-val_path',required=None, type=str,help='Validation set path')
    parser.add_argument('-n',required=False, type=str,help='Experiment name')
    parser.add_argument('-B',default=8, type=int,  help='Batch size')
    parser.add_argument('-e',default=100, type=int, help='Number of epochs')
    parser.add_argument('-w', required=False, type=bool, help= 'True if wandb initilization is required')
    parser.add_argument('-l',default=100, type=float, help='Lambda, L1 regularization')
    parser.add_argument('-p',default=34, type=int, help='Patch size of the discriminator')
    parser.add_argument('-s',required=False,default=None, type=bool, help='If smooth_label is required')
    parser.add_argument('-pad',required=False,default=1, type=int, help='Padding at Discriminator')
    parser.add_argument('-g',default=200, type=float, help='Gamma, rmse loss')
    parser.add_argument('-mu_fm_loss',default=20, type=float, help='Mu,feature matching loss')
    parser.add_argument('-lr',default=2e-4, type=float, help='learning_rate')
    parser.add_argument('-vd',default=0.001, type=float, help='upsilon discriminator, L2 regularization')
    parser.add_argument('-vg',default=0.001, type=float, help='upsilon generator, L2 regularization')

    
    args = parser.parse_args()

    epochs_check = 0               

    main(args,epochs_check)

    
    
