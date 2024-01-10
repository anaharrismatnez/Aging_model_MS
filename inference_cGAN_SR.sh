#!/bin/bash

# TRAINING PIPELINE (MS-DATASET)

eval "$(conda shell.bash hook)"

test_set=path_2_test_set
cgan_model=path_2_cgan_model
sr_model=path_2_SR_model

conda activate base # SPECIFY ENVIRONMENT OR ENVIRONMENT'S PATH

python cGAN/inference.py -data_path $test_set -model_path $cgan_model -m best_model_generator -o LR_MS
python utils/prepare_SR_data.py -input_dir "$cgan_model"/LR_MS -out_dir $sr_model/LR_MS_data -data_path $test_set

conda deactivate
conda activate supermri # SPECIFY ENVIRONMENT OR ENVIRONMENT'S PATH

python clinical-super-mri/predict.py --input_dir "$sr_model"/LR_MS_data/ --output_dir "$sr_model"/SR_MS --model_dir $sr_model

python utils/npy_2_nii.py -source $sr_model/SR_MS -data_path $test_set 

conda deactivate

