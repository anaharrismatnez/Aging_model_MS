# Ana Harris 13/02/2023
# We introduce basal MRI + delta condition (MS dataset)
# This script saves at specified folder, LR images in .npy format.

import os
import sys
main_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_folder)

import torch
from model_pix2pix import *
import argparse
from time import process_time
from utils.utils_image import *
from dataset import *
from torch.utils.data import DataLoader
import pickle

def prediction():
    print('Inference:')

    for i,data in enumerate(dataloader):
        
        basal,filename,delta = data

        filename = str(filename).split("'")[1]

        delta = delta.to(device)
        basal = basal.unsqueeze(0).to(device).float()

        fake = G(basal,delta)

        print(filename)

        img = fake.detach().squeeze().squeeze().cpu().numpy()

        np.save(f'{results_path}/{filename}.npy',img)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-model_path',required=True, type=str,help='Path to the folder that contains the model')
    parser.add_argument('-m',required=True, default='best_model', type=str,help='Model name. Default=best_model')
    parser.add_argument('-o',default='imgs_best_model', type=str,help='Output folder name. Default=imgs_best_model')
    parser.add_argument('-data_path',required=True, type=str,help='Dataset path')

    t_start = process_time() 

    args = parser.parse_args()
    
    model_path = args.m+'/checkpoints'

    results_path = args.m+'/'+args.o

    if os.path.exists(results_path) == False:
        os.mkdir(results_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'

    if args.data_path.endswith('.pkl'):
        data = pickle.load(open(args.data_path,'rb'))
    else:
        data = get_data(args.data_path)
    test_dataset = dataset(data,mode='test') 
    dataloader = DataLoader(test_dataset,shuffle=False)

    G = Generator(1,1,nf=64,aux_classes=1).to(device)

    G_checkpoint = torch.load(model_path+'/'+args.e)
    G.load_state_dict(G_checkpoint['model_state_dict'])

    prediction()

    t_finish = process_time() 
    print(f'Done in {t_finish-t_start} s')