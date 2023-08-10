- cGAN + SR pipelines are provided in `training_cGAN_SR.sh` and `inference_cGAN_SR.sh`.
- The training of cGAN is performed with the following command:
  ```
  python cGAN/main.py -h 
  usage: main.py [-h] -d D [-n N] [-B B] [-e E] [-w W] [-l L] [-p P] [-s S] [-pad PAD] [-g G] [-mu_fm_loss MU_FM_LOSS] [-lr LR] [-vd VD] [-vg VG]

  optional arguments:
    -h, --help            show this help message and exit
    -d D                  input folder
    -n N                  Experiment name
    -B B                  Batch size
    -e E                  Number of epochs
    -w W                  True if wandb initilization is required
    -l L                  Lambda, L1 regularization
    -p P                  Patch size of the discriminator
    -s S                  If smooth_label is required
    -pad PAD              Padding at Discriminator
    -g G                  Gamma, rmse loss
    -mu_fm_loss MU_FM_LOSS
                          Mu,feature matching loss
    -lr LR                learning_rate
    -vd VD                upsilon discriminator, L2 regularization
    -vg VG                upsilon generator, L2 regularization
    ```
- Inference of cGAN is performed with the following command:
  ```
  python cGAN/inference.py -h
  usage: inference.py [-h] -m MODEL -e EPOCH [-o OUT_NAME] -d D

  optional arguments:
    -h, --help            show this help message and exit
    -m MODEL, --model MODEL
                          Model experiment
    -e EPOCH, --epoch EPOCH
                          Epoch to load model. Default=best_model
    -o OUT_NAME, --out_name OUT_NAME
                          Output folder name. Default=imgs_best_model
    -d D                  Dataset path

  ```