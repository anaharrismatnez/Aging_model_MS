#!/bin/bash

# TRAINING PIPELINE (HC-DATASET)

eval "$(conda shell.bash hook)"

dataset=path_2_HC_dataset
dataset_MS=path_2_MS_dataset
model=experiment_name

conda activate ldm # SPECIFY ENVIRONMENT OR ENVIRONMENT'S PATH

python cLDM/autoencoder.py -data_path $dataset -n $model -w True -config cLDM/configs/autoencoder.yaml -agm True
python cLDM/ldm.py -data_path $dataset -n $model -w True -config cLDM/configs/ldm.yaml -AE_model results/"$model"/autoencoder/checkpoints/best_model.pth -AE_config cLDM/configs/autoencoder.yaml

# SAMPLING PIPELINE (MS-DATASET)

python cLDM/sample_images.py -data_path $dataset_MS -model_dir results/"$model"/cLDM/checkpoints/best_model.pth -config cLDM/configs/ldm.yaml -AE_model results/"$model"/autoencoder/checkpoints/best_model.pth -AE_config cLDM/configs/autoencoder.yaml -output_dir results/"$model"/cLDM/MS

python utils/npy_2_nii.py -source results/"$model"/cLDM/MS/ -data_path $dataset_MS






