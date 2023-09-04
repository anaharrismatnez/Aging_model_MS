# Ana Harris 26/05/2023
# Dataset definition (sans, patients)
# Data is structured as follow: ../Dataset_path/patient_timepoint/r_fup.npy ---> 'r_' represents it is corregistered to basal image
#                               ../Dataset_path/patient_timepoint/basal.npy
#                               ../Dataset_path/patient_timepoint/info.json ---> patient timepoint info, delta time is included.


import os
import json
from torch.utils.data import Dataset
import numpy as np
import monai

import sys
main_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from utils.utils_image import *


def threshold_at_one(x):
    # threshold at 1
    return x > 1

def threshold_at_zero(x):
    # threshold at 0
    return x > 0

augmentation_T = transforms.Compose([
    monai.transforms.AddChannel(),
    monai.transforms.CropForeground(select_fn=threshold_at_one, margin=0),
    monai.transforms.Resize((128,128,128)),
    monai.transforms.RandRotate(prob=0.4),
    monai.transforms.RandAxisFlip(prob=0.4),
    monai.transforms.ScaleIntensity(minv=0.0, maxv=1.0),
    monai.transforms.ToTensor()
])


rest_T = transforms.Compose([
    monai.transforms.AddChannel(),
    monai.transforms.CropForeground(select_fn=threshold_at_one, margin=0),
    monai.transforms.Resize((128,128,128)),
    monai.transforms.ScaleIntensity(minv=0.0, maxv=1.0),
    monai.transforms.ToTensor()
])


def get_data_AE(
    path: str,
):
        
    data_dicts = []
    for folder in os.listdir(path):
        name = folder.split('_')[1]+'_'+folder.split('_')[2]
        if folder.startswith('1_'):
            data_dicts.append(
                {
                    "scan": f"{path}/{folder}/{name}.npy",
                }
            )
        else:
            data_dicts.append(
                {
                    "scan": f"{path}/{folder}/r_{name}.npy",
                }
            )

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts

def get_data(
    path: str,
):
        
    data_dicts = []
    for folder in os.listdir(path):
        name = folder.split('_')[1]+'_'+folder.split('_')[2]
        delta = json.load(open(os.path.join(path,folder,'info.json'),'r'))['delta']
        shape = json.load(open(os.path.join(path,folder,'info.json'),'r'))['shape']
        if delta != 0:
            if folder.startswith('1_'):
                data_dicts.append(
                    {
                        "fup": f"{path}/{folder}/{name}.npy",
                        "basal": f"{path}/{folder}/{name}.npy",
                        "delta": delta,
                        "shape" : shape
                    }
                )
            else:
                data_dicts.append(
                    {
                        "fup": f"{path}/{folder}/r_{name}.npy",
                        "basal": f"{path}/{folder}/{name}.npy",
                        "delta": delta,
                        "shape" : shape
                    }
                )

        
    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts


class train_dataset(Dataset):
    def __init__(self,data : list,model : str, transform = None):
        self.data = data
        self.model = model
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.model == 'autoencoder':
            scan = np.load(self.data[index]['scan'])

            if self.transform:
                scan = self.transform(scan)
            else:
                scan = rest_T(scan)
                
            return scan

        else:
            fup = np.load(self.data[index]['fup'])
            basal = np.load(self.data[index]['basal'])

            if self.transform:
                fup = self.transform(fup)
                basal = self.transform(basal)

            else:
                fup = data_transformation(fup,128,(0,1)) 
                basal = data_transformation(basal,128,(0,1)) 


            delta = torch.tensor(float(self.data[index]['delta']))

            return fup,basal,delta


class test_dataset(Dataset):
    def __init__(self,data : list, transform = None):
        self.data = data
        self.transform = transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        basal = np.load(self.data[index]['basal'])
        filename = self.data[index]['basal'].split('/')[-2]
        shape = self.data[index]['shape']

        if self.transform:
            basal = self.transform(basal)
        else:
            basal = data_transformation(basal,128,(0,1))    


        delta = torch.tensor(float(self.data[index]['delta']))

        return basal,delta,filename,shape



