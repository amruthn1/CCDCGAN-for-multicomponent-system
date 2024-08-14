# CCDCGAN-for-multicomponent-system
The purpose of this code is to extend the functionality of the codebase by `T. Long, et al. "Constrained crystals deep convolutional generative adversarial network for the inverse design of crystal structures." npj Computational Materials 7.1 (2021): 1-7` in order to design new crystal structures of multicomponent systems. Please feel free to contact the conresponding author Prof. Hongbin Zhang (hzhang@tmm.tu-darmstadt.de) or Teng Long (tenglong@tmm.tu-darmstadt.de) for discussions on the **original** codebase.

The original codebase is linked [here](https://github.com/TengLong1993/CCDCGAN-for-single-system/tree/main).

## Preparation of the environment
To prepare the conda environment, use either the env_metal.yml, env.yml, or env_cuda.yml depending on your system and create the conda environment using:
```conda env create -f [CHOSEN_ENV].yml```

## Instructions of running
1. Make sure your data is provided in a CSV file with a CIF column including all of the CIF data

2. Edit config.json and input your Materials Project API key and the path to your CSV file

3. Prepare the CSV file by first running ```python create_training_data.py``` in the terminal.

4. Train the model first by typing ```python train_GAN.py``` in the terminal. 

5. Once completed generate new structures by typing ```python generate_new_structure.py``` in the terminal.

6. To test the feasibility of generated materials, type ```python test_feasibility.py``` in the terminal.

## Notes

1. Please note that at least 100GB of storage is required during the training process.

2. You may have to edit some values in the reconstruction function in the data_transformation.py file and the loss thresholds in the sites_autoencoder.py and lattice_autoencoder.py
