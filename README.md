# Generative Aging models for MS brain atrophy quantification.

Code for the Master Thesis: "Generation of synthetic longitudinal magnetic resonance images of subjects with multiple sclerosis: Assessment of brain atrophy and their clinical impact".


### Installation 
- Clone this repository:
  ```
  git clone https://github.com/anaharrismatnez/Aging_model_MS.git
  ```
- The [clinical-super-mri](https://github.com/bryanlimy/clinical-super-mri/tree/main) repository was used to train the Super Resolution (SR) Network. If you want the repository with all submodules, clone it with the following command:
  ```
  git clone https://github.com/bryanlimy/clinical-super-mri.git
  ```
- It is highly recommended to create a new environment. The required packages can be found in the file `requirements.txt`. For the SR network see how to create the environment in [clinical-super-mri](https://github.com/bryanlimy/clinical-super-mri/tree/main).


### Dataset

* Dataset should be structured as follows:
```
dataset/
    1_MRI_001/
        MRI_001.npy
        MRI_001_mask.npy
        info.json
        ...
    2_MRI_001/
        MRI_001.npy
        MRI_001_mask.npy
        r_MRI_001.npy
        info.json
        ...
    3_MRI_001/
        MRI_001.npy
        MRI_001_mask.npy
        r_MRI_001.npy
        info.json
        ...
  ```

- Where t_MRI_001 corresponds to each time point of patient 001, being t=1 the baseline scan, "MRI" the dataset specification, and "001" the subject identifier. As the training will be per pairs it is necessary at t > 1 to specify the follow-up scan by writing "r_" before the patient's name, meaning registered image. 
- The image "MRI_001.npy" should be skull-stripped and the "MRI_001_mask.npy" file corresponds to the binary mask.
- Info.json has to contain at least the following entries about patient scan:
  - info['shape'] = [x,y,z]
  - info['delta'] = delta value between follow-up and baseline image
- Additionally info:
  - info['baseline'] = baseline acquisition date
  - info['follow-up'] = follow-up acquisition date
  - info['difference'] = info['follow-up'] - info['baseline'] (in months)
  - info['age'] = patient's age at follow-up acquisition date

* To convert .nii files into .npy, or viceversa use the following commands:
  ```
  python utils/nii_2_npy.py -source dataset -output_dir data_npy
  ```
  ```
  python utils/npy_2_nii.py -source data_npy -data_path dataset 
  optional arguments:
  -resize
        True if .npy files are obtained from cGAN network output.

  ```
  `npy_2_nii.py` automatically saves the niftis files at folder: data_npy_niftis. 
### Models 
- [cGAN](https://github.com/anaharrismatnez/Aging_model_MS/tree/main/cGAN) + SR. Training and inference pipelines: `training_cGAN_SR.sh` and `inference_cGAN_SR.sh` 
- [cLDM](https://github.com/anaharrismatnez/Aging_model_MS/tree/main/cLDM). Training pipeline: `training_cLDM.sh`.
  
### Monitoring and Visualization 
- cGAN and cLDM monitor training through wandb via anonymus mode. 
- `clinical-super-mri` repository monitors the training via Tensorboard (check clinical-super-mri).


