# Automatic molecule design by optimizing the data-driven binary latent space via a GPU-based Ising machine

<img src="https://github.com/tsudalab/bVAE-IM/blob/main/Overview.pdf" width="600">


# Requirements
amplify==0.9.1  
joblib==1.1.0  
matplotlib==3.5.2  
networkx==2.6.3  
numexpr==2.8.1  
numpy==1.21.5  
rdkit==2022.9.5  
scikit_learn==1.2.1  
scipy==1.7.3  
torch==1.11.0
tqdm==4.64.0

# Quick Start

## Code for Accelerated Training
This repository contains the Python 3 implementation of the new Fast Junction Tree Variational Autoencoder code.

* `bvae/` contains codes for binary VAE training. Please refer to `model/README.md` for details.
* `model/` contains codes for model implementation.
* `im/` contains codes for optimizing latent binary molecular space via an Ising machine. Please refer to `im/READE.md` for details.

# Contact
Zetian Mao (zmao@g.ecc.u-tokyo.ac.jp)
Cite this code: [![DOI](https://zenodo.org/badge/608057945.svg)](https://zenodo.org/badge/latestdoi/608057945)