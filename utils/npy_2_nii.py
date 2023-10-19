import os
import sys
main_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_folder)

import numpy as np
from utils.utils_image import *
import argparse



def npy2nii():

    for filename in os.listdir(args.source):
        print(filename)
        if filename.endswith('.npy'):
            folder_name = filename.split('.npy')[0]
            shape,affine = check_nifti_data(os.path.join(args.data_path,folder_name))

            new_name = filename.replace('.npy','.nii.gz')
            img = np.load(os.path.join(args.source,filename))

            save_nifti(img,affine,None,f'{out_dir}/{new_name}')   



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-source',required=True, type=str,help='Folder with .npy files (all files in 1 folder, result from inference)')
    parser.add_argument('-data_path',required=True,type=str,help='Dataset path')

    args = parser.parse_args()
    
    path_source = args.source.split('/')

    if path_source[-1] == '':
        path_source[-2] = f'{path_source[-2]}_niftis'
        out_dir = ('/').join(path_source)
    else:
        out_dir = f'{args.source}_niftis'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    npy2nii()