#!/bin/bash

# TRAINING PIPELINE (HC-DATASET)

eval "$(conda shell.bash hook)"

dataset=path_2_HC_dataset
dataset_MS=path_2_MS_dataset
experiment=experiment_name

python cLDM/autoencoder.py -d $dataset -n $experiment -w True -config cLDM/configs/autoencoder.yaml -agm True

python cLDM/ldm.py -d $dataset -n $experiment -w True -config cLDM/configs/ldm.yaml -AE_model cLDM/results/"$experiment"/autoencoder/checkpoints/best_model.pth -AE_config cLDM/configs/autoencoder.yaml

# SAMPLING PIPELINE (MS-DATASET)

python cLDM/sample_images.py -d $dataset_MS -model_dir cLDM/results/"$experiment"/ldm/checkpoints/best_model.pth -config cLDM/configs/ldm.yaml -AE_model cLDM/results/"$experiment"/autoencoder/checkpoints/best_model.pth -AE_config cLDM/configs/autoencoder.yaml -output_dir cLDM/results/"$experiment"/ldm/MS

python utils/npy_2_nii.py --source cLDM/results/"$experiment"/ldm/MS/LR/ --data_path $dataset_MS






