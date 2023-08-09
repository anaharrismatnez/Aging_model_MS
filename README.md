# Generative Aging models for MS brain atrophy quantification.

Code for the Master Thesis: <span style="color: blue"> "Generation of synthetic longitudinal magnetic resonance images of subjects with multiple sclerosis: Assessment of brain atrophy and their clinical impact" </span>

PONER UN GIF???

### Installation 
añadir lo del supermri
requirements

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

- Where {t}_MRI_001 corresponds to each time point of patient 001, being t=1 the baseline scan. As the training will be per pairs it is necessary at t > 1 to specify the follow-up scan by writing "r_" before the patient's name, meaning registered image.
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
  python utils/npy_2_nii.py --source data_npy --data_path dataset --SR True
  ```

  python utils/npy_2_nii.py automatically saves the niftis files at folder: data_npy_niftis. Command --SR specifies if the .npy images are generated from the clinical-super-mri (Super Resolution Network).

### Models 
- enlaces de cada modelo y allí especificar readme con comands
  
### MONITORING 
decir wandb

