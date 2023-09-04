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

def prediction():
    print('Inference:')
    for i,data in enumerate(dataloader):
        
        basal,filename,delta = data
        delta = delta.to(device)
        basal = basal.unsqueeze(0).to(device).float()
        fake = G(basal,delta)

        filename = str(filename).split("'")[1]

        print(filename)
        
        img = fake.detach().squeeze().squeeze().cpu().numpy()

        np.save(f'{results_path}/{filename}.npy',img)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model',required=True, type=str,help='Model experiment')
    parser.add_argument('-e','--epoch',required=True, default='best_model', type=str,help='Epoch to load model. Default=best_model')
    parser.add_argument('-o','--out_name',default='imgs_best_model', type=str,help='Output folder name. Default=imgs_best_model')
    parser.add_argument('-data_path',required=True, type=str,help='Dataset path')

    t_start = process_time() 

    args = parser.parse_args()
    
    abs_path = os.path.dirname(os.path.abspath(__file__))

    model_path = abs_path+'/results/'+args.model+'/training_cGAN'


    results_path = model_path+'/'+args.out_name

    model_path = model_path+'/checkpoints'

    if os.path.exists(results_path) == False:
        os.mkdir(results_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = get_data(args.d)
    test_dataset = dataset(data,mode='test') 
    dataloader = DataLoader(test_dataset, shuffle=True)

    G = FiLMed_Generator(1,1,nf=64,aux_classes=1).to(device)

    G_checkpoint = torch.load(model_path+'/'+args.epoch+'_generator')
    G.load_state_dict(G_checkpoint['model_state_dict'])

    prediction()

    t_finish = process_time() 
    print(f'Done in {t_finish-t_start} s')