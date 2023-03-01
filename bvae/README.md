# Training of Junction Tree binary VAE (JTbVAE)

## Training
Step 1: Preprocess the data:
```
python preprocess.py --train ../data/train.txt --split 100 --jobs 40 --output ./train-processed
```
This script will preprocess the training data (subgraph enumeration & tree decomposition), and save results into a list of files. We suggest you to use small value for `--split` if you are working with smaller datasets.

Step 2: Train VAE model with KL annealing.
```
python vae_train.py --train train-processed --vocab ../data/vocab.txt --save_dir vae_model/
```
Required Options:

`--train` sets the path to the processed training data.

`--vocab` sets the path to the extracted vocabulary.

`--save_dir` sets the path to save models during training.

Default Options:

`--binary_size 300` sets the dimensionality of binary latent space to 300.

`--beta 0.001` means to set KL regularization weight (beta) initially to be 0.001.

`--init_temp 1.0` means that the temperature $\tau_0$ is initally set as 1.0 in Gumbel softmax $y_i=\frac{exp((log(\pi_i)+g_i)/\tau)}{\sum_{j=1}^kexp((log(\pi_i)+g_i)/\tau)}$.

`--temp_anneal_rate 0.0001 --min_temp 0.4` means that the temperature will be updated by $\tau'=max(0.4, \tau\times exp(-0.0001t))$, where $t$ is the current step during training.

Note that these hyperparameters are used in the present research, and may not be the best training strategy. Further adjustment is welcomed.

The 300-dimensional JTbVAE model employed in the research  is available in `"./vae_model/model-dim300"`.