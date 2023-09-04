- cLDM pipelines is provided in `training_cLDM.sh` and `inference_cLDM.sh`.
- The training of the AE is performed with the following command:
  ```
  python cLDM/autoencoder.py -h 
  usage: autoencoder.py [-h] -data_path DATA_PATH [-n N] [-B B] [-e E] [-w W] -config
                      CONFIG [-agm AGM]

  optional arguments:
    -h, --help            show this help message and exit
    -data_path DATA_PATH  Data path
    -n N                  Experiment name
    -B B                  Batch size
    -e E                  Number of epochs
    -w W                  True if wandb initilization is required
    -config CONFIG        Config file
    -agm AGM              If data augmentation is required

    ```
- The training of the cLDM is performed with the following command:
  ```
  python cLDM/ldm.py -h
  usage: ldm.py [-h] -ddata_path DDATA_PATH [-n N] [-B B] [-e E] [-e0 E0] [-w W]
              -config CONFIG -AE_model AE_MODEL -AE_config AE_CONFIG

  optional arguments:
    -h, --help            show this help message and exit
    -ddata_path DDATA_PATH
                          Data path
    -n N                  Experiment name
    -B B                  Batch size
    -e E                  Number of epochs
    -e0 E0                Epoch to load model
    -w W                  True if wandb initilization is required
    -config CONFIG        Config file
    -AE_model AE_MODEL    Path to autoencoder model.
    -AE_config AE_CONFIG  Path to autoencoder model.

  ```
- To get sample images use this command:
    ```
  python cLDM/sample_images.py -h
  usage: sample_images.py [-h] -data_path DATA_PATH -model_dir MODEL_DIR -AE_model
                        AE_MODEL -config CONFIG -AE_config AE_CONFIG
                        [-guidance_scale GUIDANCE_SCALE] [-x_size X_SIZE]
                        [-y_size Y_SIZE] [-z_size Z_SIZE]
                        [-num_inference_steps NUM_INFERENCE_STEPS] -output_dir
                        OUTPUT_DIR

  optional arguments:
    -h, --help            show this help message and exit
    -data_path DATA_PATH  Data path
    -model_dir MODEL_DIR  Path to the .pth model from the diffusion model.
    -AE_model AE_MODEL    Path to the .pth model from the stage1.
    -config CONFIG        Config file
    -AE_config AE_CONFIG  Config file
    -guidance_scale GUIDANCE_SCALE
    -x_size X_SIZE        Latent space x size.
    -y_size Y_SIZE        Latent space y size.
    -z_size Z_SIZE        Latent space z size.
    -num_inference_steps NUM_INFERENCE_STEPS
    -output_dir OUTPUT_DIR
                          Output directory

  ```