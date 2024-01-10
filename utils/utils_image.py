from skimage.transform import resize
import nibabel as nib
from torchvision import transforms
import numpy as np
import torch
import os
import json

def read_nifti(path,transforms=None):
    # In: volume file in nifti format
    # Out: Array image
    img = nib.load(path)
    if transforms:
        return img.get_fdata(), img.affine,img.header
    else:
        return img.get_fdata()


def img_normalization(image,range=(-1,1)):
    if range == (-1,1):
        normalized = 2 * (image - np.min(image)) / (np.max(image) - np.min(image)) - 1
    else:
        normalized = (image - np.min(image)) / (np.max(image) - np.min(image)) 
    return normalized



def data_transformation(image,img_size,range=(-1,1)):
    transform = transforms.Compose([transforms.ToTensor()])
    if img_size:
        image = resize(image,(img_size, img_size, img_size))
    image = img_normalization(image,range)
    return transform(image)

def data_transformation_numpy(image,img_size,range=(-1,1)):
    if img_size:
        image = resize(image,(img_size))
    if range:
        image = img_normalization(image,range)
    
    return image

def save_nifti(image,affine,header,outputImageFileName):
    # Function to save generated images in nifti format
    out = nib.Nifti1Image(image, affine, header)
    nib.save(out, outputImageFileName)    

def generate_mask(image,l=-1):
    mask = torch.ne(image,l)
    mask = mask.type(torch.float32)
    return mask

def check_nifti_data(folder_path):
    delta = json.load(open(os.path.join(folder_path,'info.json'),'r'))
    shape = delta['shape']
    affine = np.load(os.path.join(folder_path,'affine.npy'))
    
    return shape,affine