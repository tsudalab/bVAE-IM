# Chemical Design with GPU-based Ising Machine

## Details
Details are available at our published paper [https://doi.org/10.1039/D3DD00047H](https://doi.org/10.1039/D3DD00047H).

## Abstract

Ising machines
are hardware-assisted discrete optimizers
that often outperform purely software-based optimization.
They are implemented, e.g., with superconducting qubits, ASICs or GPUs.
In this paper, we show how Ising machines can be leveraged to gain
efficiency improvements in automatic molecules design. 
To this aim, we construct a graph-based binary variational autoencoder
to obtain discrete latent vectors,
train a factorization machine as a surrogate model,
and optimize it with an Ising machine.
In comparison to Bayesian optimization in a continuous latent space,
our method performed better in three benchmarking problems.
Two types of Ising machines, qubit-based D-Wave quantum annealer
and GPU-based Fixstars [Amplify](https://amplify.fixstars.com/en/), are compared to observe that
GPU-based one scales better and more suitable for molecule generation.
Our results show that GPU-based Ising machines have the potential
to empower deep-learning-based materials design.

<p align="center">
  <img src="https://github.com/tsudalab/bVAE-IM/blob/main/overview.png" width="600">
</p>

The implementation of binary VAE is based on the work [Junction Tree Variational Autoencoder for Molecular Graph Generation.](https://arxiv.org/pdf/1802.04364.pdf)

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

* `data/` contains codes for randomly selecting labeled data. Please refer to `data/README.md` for details.
* `train/` contains codes for binary VAE training. Please refer to `train/README.md` for details.
* `bJTVAE/` contains codes for the binary VAE model implementation.
* `im/` contains codes for optimizing latent binary molecular space via an Ising machine. Please refer to `im/README.md` for details.

# Contact
Zetian Mao (zmao@g.ecc.u-tokyo.ac.jp)\
Department of Computational Biology and Medical Science\
The University of Tokyo\
Cite this code: [![DOI](https://zenodo.org/badge/608057945.svg)](https://zenodo.org/badge/latestdoi/608057945)
