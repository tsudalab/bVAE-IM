# Data Generation

## Unlabelled data for training the bJTVAE

We used the 250k version ZINC data, same as the original junction tree VAE work, to train the binary junction tree VAE.

The data source is available [here](https://github.com/wengong-jin/icml18-jtnn/tree/master/data/zinc).

## Labelled data for training the factorization machine

We employed the factorization machine as the surrogate model for target property regression.

Three target properties, penalized logp, topological polar surface area (TPSA), and a multi-objective property (GSK3-$\beta$+JSN3+SA+QED), are optimized
to evaluate our approach.
