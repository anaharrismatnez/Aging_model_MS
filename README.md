# Generative Aging models for MS brain atrophy quantification.

Code for the Master Thesis: "Generation of synthetic longitudinal magnetic resonance images of subjects with multiple sclerosis: Assessment of brain atrophy and their clinical impact".
AÑADIR + DESCRIPCION
PONER UN GIF???

### Installation 
- Clone this repository:
  ```
  git clone https://github.com/anaharrismatnez/Aging_model_MS.git
  ```
- The [clinical-super-mri](https://github.com/bryanlimy/clinical-super-mri/tree/main) repository was used to train the Super Resolution (SR) Network. If you want the repository with all submodules, clone it with the following command:
  ```
  git clone https://github.com/bryanlimy/clinical-super-mri.git
  ```
- It is highly recommended to create a new environment for each of the models to be used. The required packages can be found in the file `requirements.txt`. For the SR network see how to create the environment in [clinical-super-mri](https://github.com/bryanlimy/clinical-super-mri/tree/main).


### Dataset

* Dataset should be structured as follows:
```
dataset/
    1_MRI_001/
        MRI_001.npy
        info.json
        ...
    2_MRI_001/
        MRI_001.npy
        r_MRI_001.npy
        info.json
        ...
    3_MRI_001/
        MRI_001.npy
        r_MRI_001.npy
        info.json
        ...
  ```

- Where t_MRI_001 corresponds to each time point of patient 001, being t=1 the baseline scan. As the training will be per pairs it is necessary at t > 1 to specify the follow-up scan by writing "r_" before the patient's name, meaning registered image.
- Info.json contains all information about patient scan:
  - info['shape'] = [x,y,z]
  - info['delta'] = delta value between follow-up and baseline image
- Additionally info:
  - info['baseline'] = baseline acquisition date
  - info['follow-up'] = follow-up acquisition date
  - info['difference'] = info['follow-up'] - info['baseline'] (in months)
  - info['age'] = patient's age at follow-up acquisition date

* To convert .nii files into .npy, or viceversa use the following commands:
  ```
  python utils/nii_2_npy.py --source dataset --output_dir data_npy
  ```
  ```
  python utils/npy_2_nii.py --source data_npy --data_path dataset 
  optional arguments:
  --SR
        True if .npy files are obtained from SR network output

  ```
  `npy_2_nii.py` automatically saves the niftis files at folder: data_npy_niftis. 
### Models 
- [cGAN + SR](https://github.com/anaharrismatnez/Aging_model_MS/tree/main/cGAN) The training and inference pipelines can be found in the files: `training_cGAN_SR.sh` and `inference_cGAN_SR.sh` 
- 
  
### Monitoring and Visualization 
- cGAN and cLDM monitor training through wandb via anonymus mode. 
- `clinical-super-mri` repository monitors the training via Tensorboard (check clinical-super-mri).


