#!/bin/bash

# TRAINING PIPELINE (MS-DATASET)

eval "$(conda shell.bash hook)"

dataset=path_2_MS_dataset
model=best_model_cGAN

conda activate base

python cGAN/inference.py -data_path $dataset -m $model -e best_model -o LR_MS
python utils/prepare_SR_data.py -input_dir results/"$model"/training_cGAN/LR_MS -out_dir results/"$model"/training_SR/MS_data -data_path $dataset

conda deactivate
conda activate supermri

python clinical-super-mri/predict.py --input_dir results/"$model"/training_SR/MS_data/ --output_dir results/"$model"/training_SR/MS_SR --model_dir results/"$model"/training_SR/ 

python utils/npy_2_nii.py -source results/$model/training_SR/MS_SR -data_path $dataset 

conda deactivate

