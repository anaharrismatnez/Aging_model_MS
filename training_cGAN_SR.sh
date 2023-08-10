#!/bin/bash

# TRAINING PIPELINE (HC-DATASET)

eval "$(conda shell.bash hook)"

dataset=path_2_HC_dataset
model=best_model_cGAN

conda activate cGAN

python cGAN/main.py -d $dataset -e 150 -w True -n $model
python cGAN/inference.py -d $dataset -m $model -e best_model -o LR_HC
python utils/prepare_SR_data.py --input_dir cGAN/results/"$model"/training_cGAN/LR_HC --out_dir cGAN/results/"$model"/training_SR/training_data --HR True --basals True

conda deactivate
conda activate supermri

python clinical-super-mri/train.py --input_dir cGAN/results/"$model"/training_SR/training_data/ --output_dir /cGAN/results/"$model"/training_SR/ --extension npy --patch_size 64 --n_patches 5000 --epochs 100 --normalization instancenorm --num_filters 128
python clinical-super-mri/predict.py --input_dir cGAN/results/"$model"/training_SR/training_data/ --output_dir /cGAN/results/"$model"/training_SR/HC_SR --model_dir cGAN/results/"$model"/training_SR/

python utils/npy_2_nii.py --source cGAN/results/$model/training_SR/HC_SR --data_path $dataset --SR True

conda deactivate


