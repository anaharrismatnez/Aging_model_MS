# Ana Harris 06/04/2023
# File to prepare data for training Super Resolution (SR) Network, as required in: https://github.com/bryanlimy/clinical-super-mri/tree/main

import os
import sys
main_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_folder)

from utils.util import move_HR_files,move_LR_files
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_dir',required=True,type=str,help='Results obtained from cGAN pipeline.Folder with LR images to use.')
    parser.add_argument('-out_dir',required=True,type=str,help='Folder to save LR and HR (if needed).')
    parser.add_argument('-data_path',required=True,type=str,help='Dataset path.')
    parser.add_argument('-HR',required=False,type=bool,help='If we want to move HR images (necessary for training SR).')
    parser.add_argument('-basals',required=False,type=bool,help='If we want to move baseline images (train only with basals).')

    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir) 

    move_LR_files(args.input_dir,args.out_dir,args.data_path)

    if args.HR:
        move_HR_files(args.data_path,args.out_dir)
    