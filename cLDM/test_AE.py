import argparse
import os
from omegaconf import OmegaConf
import torch
import torch.optim as optim
import sys

sys.path.append('~Ana/code/brain_ldm/GenerativeModels/')

from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL
from generative.networks.nets.patchgan_discriminator import PatchDiscriminator
from generative.losses.adversarial_loss import PatchAdversarialLoss
from monai.utils import set_determinism
from datetime import date
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from utils.dataset import *
from collections import OrderedDict
from utils.util import * 
import pickle



torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MODEL CONFIGURATION
config = OmegaConf.load('configs/autoencoder_test9.yaml')
model = AutoencoderKL(**config["stage1"]["params"]).to(device)
D = PatchDiscriminator(**config["discriminator"]["params"]).to(device)
perceptual_loss = PerceptualLoss(**config["perceptual_network"]["params"]).to(device)

G_optimizer = optim.Adam(model.parameters(), lr=config["stage1"]["base_lr"])
D_optimizer = optim.Adam(D.parameters(), lr=config["stage1"]["disc_lr"])


out_dir = '/home/extop/Ana/code/brain_ldm/results/test9_agm/autoencoder/final_best'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

checkpoint = torch.load(f'/home/extop/Ana/code/brain_ldm/results/test9_agm/autoencoder/checkpoints/final_model.pth')
model.load_state_dict(checkpoint['model'])
print('Model loaded from checkpoint at epoch:',checkpoint['epoch'])  

model.eval()

data = pickle.load(open('/home/extop/Ana/Datasets/Niftis_sans_Workspaces_biogen/numpys_20_AE.pkl','rb'))
new_data = data[:10]

for i in range(len(new_data)):
    img = np.load(new_data[i]['scan'])
    img = rest_T(img)
    img = img.unsqueeze(0).to(device).float()

    with autocast(enabled=True):
        with torch.no_grad():
            reconstruction, z_mu, z_sigma = model(x=img)

    fig = get_figure_AE(img,reconstruction,display=False)
    fig.savefig(f'{out_dir}/{i}.png')

print('done')