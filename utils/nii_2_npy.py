import os
import sys
main_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_folder)

import numpy as np
from utils.utils_image import *
import shutil
import argparse
import json

def nii2npy(args):

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for folder in os.listdir(args.source):
        if not os.path.exists(os.path.join(args.output_dir,folder)):
            os.makedirs(os.path.join(args.output_dir,folder))

        name = folder.split('_')[-2]+'_'+folder.split('_')[-1]
        #name = folder

        for filename in os.listdir(os.path.join(args.source,folder)):
            new_name = filename.split('.nii.gz')[0]+'.npy'
            if name in filename:
                if not 'r_' in filename:
                    img,affine,header = read_nifti(os.path.join(args.source,folder,filename),meta=True) # Automatically saves affine and header from basal image.
                    np.save(os.path.join(args.output_dir,folder,'affine.npy'),affine)
                    #np.save(os.path.join(args.output_dir,folder,'header.npy'),header)

                else:
                    img = read_nifti(os.path.join(args.source,folder,filename))

                np.save(os.path.join(args.output_dir,folder,new_name),img)
                print(new_name,':',img.shape)

        info = json.load(open(os.path.join(args.source,folder,'info.json'),'r'))
        info['shape'] = img.shape
        with open(os.path.join(args.source,folder,'info.json'),'w') as h:
            json.dump(info,h)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-source',required=True, type=str,help='Dataset folder (grouped by patient folders)')
    parser.add_argument('-output_dir',required=True,type=str,help='Folder to save .npy files')

    args = parser.parse_args()

    nii2npy(args)
