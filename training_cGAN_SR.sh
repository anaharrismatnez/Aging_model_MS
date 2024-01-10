#!/bin/bash

# TRAINING PIPELINE (HC-DATASET)

eval "$(conda shell.bash hook)"

dataset_HC=path_2_HC_dataset
dataset_MS=path_2_MS_dataset
validation_set=path_2_validation_set
model_HC=aging_model
model_MS=MS_model

conda activate base # SPECIFY ENVIRONMENT OR ENVIRONMENT'S PATH

python cGAN/main.py -data_path $dataset_HC -val_path $validation_set -e 200 -w True -n $model_HC -B 4
python cGAN/main.py -data_path $dataset_MS -val_path $validation_set -e 200 -w True -B 4 -n $model_MS

python cGAN/inference.py -data_path $dataset_MS -model_path results/$model_MS -m best_model_generator -o LR_training

python utils/prepare_SR_data.py -input_dir results/"$model_MS"/LR_training -out_dir results/training_SR_MS/training_data -HR True -basals True -data_path $dataset_MS

python cGAN/inference.py -data_path $dataset_HC -model_path results/$model_HC -m best_model_generator -o LR_training

python utils/prepare_SR_data.py -input_dir results/"$model_HC"/LR_training -out_dir results/training_SR_HC/training_data -HR True -basals True -data_path $dataset_HC

conda deactivate
conda activate supermri # SPECIFY ENVIRONMENT OR ENVIRONMENT'S PATH

python clinical_super_mri/train.py --input_dir results/training_SR_MS/training_data/ --output_dir results/training_SR_MS/ --extension npy --patch_size 64 --n_patches 5000 --epochs 100 --normalization instancenorm --num_filters 128

python clinical_super_mri/train.py --input_dir results/training_SR_HC/training_data/ --output_dir results/training_SR_HC/ --extension npy --patch_size 64 --n_patches 5000 --epochs 100 --normalization instancenorm --num_filters 128


conda deactivate


