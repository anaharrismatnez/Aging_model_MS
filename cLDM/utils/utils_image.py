from skimage.transform import resize
import nibabel as nib
from torchvision import transforms
from scipy import stats
import numpy as np
import torch
from scipy.ndimage import zoom

def read_nifti(path,meta=None):
    # In: volume file in nifti format
    # Out: Array image
    img = nib.load(path)
    if meta:
        return img.get_fdata(), img.affine,img.header
    else:
        return img.get_fdata()



def img_normalization(image,range=(-1,1)):
    if range == (-1,1):
        normalized = 2 * (image - np.min(image)) / (np.max(image) - np.min(image)) - 1
    else:
        normalized = (image - np.min(image)) / (np.max(image) - np.min(image)) 
    return normalized

def img_stadardization(image):
    normalized = (image - torch.mean(image[image != torch.min(image)])) / torch.std(image[image != torch.min(image)])
    return normalized

def data_transformation(image,img_size,range=(-1,1)):
    transform = transforms.Compose([transforms.ToTensor()])
    if img_size:
        if type(img_size) == tuple:
            image = resize(image,img_size)
        else:
            image = resize(image,(img_size, img_size, img_size))
    image = img_normalization(image,range)
    return transform(image)

def data_transformation_numpy(image,img_size,range=(-1,1)):
    if img_size:
        image = resize(image,(img_size, img_size, img_size))
    if range:
        image = img_normalization(image,range)
    
    return image

def crop_volumes(img,crop_shape=(180,220,180)):

    if not img.shape[0] < crop_shape[0]:
        start_x = (img.shape[0] - crop_shape[0]) // 2
        end_x = (start_x+crop_shape[0])
    else:
        start_x = 0
        end_x = img.shape[0]
    if not img.shape[1] < crop_shape[1]:
        start_y = (img.shape[1] - crop_shape[1]) // 2
        end_y = (start_y+crop_shape[1])
    else:
        start_y = 0
        end_y = img.shape[1]
    if not img.shape[2] < crop_shape[2]:
        start_z = (img.shape[2] - crop_shape[2]) // 2
        end_z = (start_z+crop_shape[2])
    else:
        start_z = 0
        end_z = img.shape[2]

    cropped_data = img[start_x:end_x, start_y:end_y, start_z:end_z]
    
    return cropped_data

def save_nifti(image,affine,header,outputImageFileName):
    # Function to save generated images in nifti format
    out = nib.Nifti1Image(image, affine, header)
    nib.save(out, outputImageFileName)    

def generate_mask(image,l=-1):
    mask = torch.ne(image,l)
    mask = mask.type(torch.float32)
    return mask
