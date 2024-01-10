# Ana Harris 19/01/2023
# Training dataset: basal image + delta (condition) + follow-up 
# Test dataset: basal image + delta
# Dataset folder is structured as follows: each folder corresponds to a patient time-point. For example, foldername: 1_SUBJECT Timepoint 1 of patient SUBJECT
# Within each folder we have the follow-up T1 (corregistered with basal T1) and basal T1. In the case of timepoint 1 we have only basal (since we want to use it as delta 0). 
# DELTA: delta represents the difference between time-point and basal in years.


import os
from torch.utils.data import Dataset
import sys
main_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_folder)
from utils.utils_image import *
import json
import monai


def threshold_at_zero(x):
    # threshold at 0
    return x > 0

""" transforms = monai.transforms.Compose([
    monai.transforms.EnsureChannelFirst(channel_dim='no_channel'),
    monai.transforms.CropForeground(select_fn=threshold_at_zero, margin=0),
    monai.transforms.Resize((128,128,128)),
    monai.transforms.ScaleIntensity(minv=-1.0, maxv=1.0),
    monai.transforms.ToTensor()
]) """

def get_data(
    path: str,
):
        
    data_dicts = []
    for folder in os.listdir(path):
        name = folder.split('_')[1]+'_'+folder.split('_')[2]
        info = json.load(open(os.path.join(path,folder,'info.json'),'r'))
        if info['delta'] == 0:
            data_dicts.append(
                {
                    "fup": f"{path}/{folder}/{name}.npy",
                    "basal": f"{path}/{folder}/{name}.npy",
                    "delta": info['delta']
                }
            )
        else:
            data_dicts.append(
                {
                    "fup": f"{path}/{folder}/r_{name}.npy",
                    "basal": f"{path}/{folder}/{name}.npy",
                    "delta": info['delta']
                }
            )

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts


class dataset(Dataset):
    def __init__(self,data,mode='training'):
        self.data = data
        self.transform = transforms
        self.mode = mode
        
    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        if self.mode == 'training':
            patient_name = self.data[index]['basal'].split('/')[-2]
            gt = np.load(self.data[index]['fup'])
            gt = data_transformation(gt,128)
            #gt = gt.transform(gt)
            condition = np.load(self.data[index]['basal'])
            #condition = self.transform(condition)
            condition = data_transformation(condition,128)

            delta = torch.tensor(float(self.data[index]['delta']))

            return gt,condition,patient_name,delta 

        elif self.mode == 'validation':
            patient_name = self.data[index]['basal'].split('/')[-2]
            gt = np.load(self.data[index]['fup'])
            #gt = self.transform(gt)
            gt = data_transformation(gt,128)
            condition = np.load(self.data[index]['basal'])
            condition = data_transformation(condition,128)
            #condition = self.transform(condition)

            delta = torch.tensor(float(self.data[index]['delta']))

            return gt,condition,patient_name,delta 

        else:
            patient_name = self.data[index]['basal'].split('/')[-2]  
            condition = np.load(self.data[index]['basal']) 
            #condition = self.transform(condition)
            condition = data_transformation(condition,128)
            delta = torch.tensor(float(self.data[index]['delta']))
            return condition,patient_name,delta




