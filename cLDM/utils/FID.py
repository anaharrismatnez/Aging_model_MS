""" Script to compute the Frechet Inception Distance (FID) of the samples of the LDM.

In order to measure the quality of the samples, we use the Frechet Inception Distance (FID) metric between 1200 images
from the MIMIC-CXR dataset and 1000 images from the LDM.
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchxrayvision as xrv
from generative.metrics import FIDMetric
import monai
from monai.config import print_config
from monai.utils import set_determinism
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataset import * 

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--sample_dir",default='/home/extop/Ana/code/brain_pix2pix_MS/results/FILMv4/SR_5000_instancenorm_128/remove/', help="Location of the samples to evaluate.")
    parser.add_argument("--test_dir",default='/home/extop/Ana/Datasets/Niftis_pato_SELs/remove/', help="Location of file with test ids.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of loader workers")

    args = parser.parse_args()
    return args


def main(args):
    set_determinism(seed=args.seed)
    print_config()

    samples_dir = Path(args.sample_dir)
    test_dir = Path(args.test_dir)

    # Load pretrained model
    device = torch.device("cuda")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model = model.to(device)
    model.eval()

    # Samples
    samples_datalist = []
    for sample_path in sorted(list(samples_dir.glob("*.npy"))):
        samples_datalist.append(
            {
                "scan": str(sample_path),
            }
        )
    print(f"{len(samples_datalist)} images found in {str(samples_dir)}")

    samples_ds = train_dataset(samples_datalist,model='autoencoder')

    samples_loader = DataLoader(
        samples_ds,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    samples_features = []
    for batch in tqdm(samples_loader):
        img = batch["image"]
        with torch.no_grad():
            outputs = model.features(img.to(device))
            outputs = F.adaptive_avg_pool2d(outputs, 1).squeeze(-1).squeeze(-1)  # Global average pooling

        samples_features.append(outputs.cpu())
    samples_features = torch.cat(samples_features, dim=0)

"""     # Test set
    test_datalist = []
    for test_path in sorted(list(test_dir.glob("*.npy"))):
        test_datalist.append(
            {
                "image": str(test_path),
            }
        )
    print(f"{len(test_datalist)} images found in {str(test_dir)}")

    test_ds = dataset(test_datalist,rest_T)

    test_loader = DataLoader(
        test_ds,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    test_features = []
    for batch in tqdm(test_loader):
        img = batch["image"]
        with torch.no_grad():
            outputs = model.features(img.to(device))
            outputs = F.adaptive_avg_pool2d(outputs, 1).squeeze(-1).squeeze(-1)  # Global average pooling

        test_features.append(outputs.cpu())
    test_features = torch.cat(test_features, dim=0)

    # Compute FID
    metric = FIDMetric()
    fid = metric(samples_features, test_features)

    print(f"FID: {fid:.6f}")
 """

if __name__ == "__main__":
    args = parse_args()
    main(args)