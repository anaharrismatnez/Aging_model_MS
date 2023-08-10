#!/bin/bash

# TRAINING PIPELINE (MS-DATASET)

eval "$(conda shell.bash hook)"

dataset=path_2_MS_dataset
model=best_model_cGAN

conda activate cGAN

python cGAN/inference.py -d $dataset -m $model -e best_model -o LR_MS
python utils/prepare_SR_data.py --input_dir cGAN/results/"$model"/training_cGAN/LR_MS --out_dir cGAN/results/"$model"/training_SR/MS_data 

conda deactivate
conda activate supermri

python clinical-super-mri/predict.py --input_dir cGAN/results/"$model"/training_SR/MS_data/ --output_dir /cGAN/results/"$model"/training_SR/MS_SR --model_dir cGAN/results/"$model"/training_SR/

python utils/npy_2_nii.py --source cGAN/results/$model/training_SR/MS_SR --data_path $dataset --SR True

conda deactivate

