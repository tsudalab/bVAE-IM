# Data Generation

## Unlabelled data for training the bJTVAE

We used the 250k version ZINC data, same as the original junction tree VAE work, to train the binary junction tree VAE.

The data source is available [here](https://github.com/wengong-jin/icml18-jtnn/tree/master/data/zinc).

## Labelled data for molecule optimization

We employed the factorization machine as the surrogate model for target property regression. Six target properties, penalized logp, topological polar surface area (TPSA), molecular weight (MW), number of aromatic rings ($N_{ar}$), number of rotatable bonds ($N_{rb}$), and a multi-objective property (GSK3-&beta;+JSN3+SA+QED), are optimized to test our approach in the paper.

To evaluate the extrapolation ability, we intentionally limited the property range of the training data: $\textrm{LogP}\in[-3,3]$, $\textrm{TPSA}\in[0,100]$, $\textrm{MW}\in[0,5]$, $N_{ar}\in[0,2]$, $N_{rb}\in[0,5]$, $\textrm{multi}\in[0, 0.5]$.

The 10k labelled data used in our experiment are extracted using the following code

`python gen_logp_labelled_data.py`

`python gen_tpsa_labelled_data.py`

`python gen_mw_labelled_data.py`

`python gen_aroring_labelled_data.py`

`python gen_rotbond_labelled_data.py`

`python gen_multi_labelled_data.py`
