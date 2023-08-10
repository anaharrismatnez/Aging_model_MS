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


def get_data(
    path: str,
):
        
    data_dicts = []
    for folder in os.listdir(path):
        name = folder.split('_')[1]+'_'+folder.split('_')[2]
        delta = json.load(open(os.path.join(path,folder,'info.json'),'r'))['delta']
        if folder.startswith('1_'):
            data_dicts.append(
                {
                    "fup": f"{path}/{folder}/{name}.npy",
                    "basal": f"{path}/{folder}/{name}.npy",
                    "delta": delta
                }
            )
        else:
            data_dicts.append(
                {
                    "fup": f"{path}/{folder}/r_{name}.npy",
                    "basal": f"{path}/{folder}/{name}.npy",
                    "delta": delta
                }
            )

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts


class dataset(Dataset):
    def __init__(self,data,img_size=128,mode='training'):
        self.data = data
        self.img_size = img_size
        self.mode = mode

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        if self.mode == 'training':
            patient_name = self.data[index]['basal'].split('/')[-2]
            gt = np.load(self.data[index]['fup'])
            gt = data_transformation(gt,self.img_size)
            condition = np.load(self.data[index]['basal'])
            condition = data_transformation(condition,self.img_size)

            delta = torch.tensor(float(self.data[index]['delta']))

            return gt,condition,patient_name,delta 

        else:
            patient_name = self.data[index]['basal'].split('/')[-2]  
            condition = np.load(self.data[index]['basal']) 
            condition = data_transformation(condition,self.img_size)
            delta = torch.tensor(float(self.data[index]['delta']))
            return condition,patient_name,delta






