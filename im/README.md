# Optimization on the Binary  Latent Space

## Optimization

Here gives an example for optimizing penalized LogP based on 10k labeled data.

```
python bVAE-IM_LogP.py --model ../bvae/vae_model/model-dim300 --vocab ../data/zinc/vocab.txt --dim 300 --smiles ../data/opt/plogp_train_smiles10k.npy --prop ../data/opt/plogp_train_props10k.npy --output ./results --cache ./cache --token xxxxxxxxxx --patience 300 --target max --seed 1 --device cuda --client amplify
```

Required Options:

`--model` sets the path to the saved model that builds the binary space.

`--vocab` sets the path to the extracted vocabulary.

`--dim300` sets the dimensionality for the binary space.

`--smiles` loads the prepared smiles list for training the factorization machine. The file is saved by Numpy.

`--prop` loads the property values of correponding smiles, saved in Numpy.

`--results` sets the directory that saves the output results.

`--cache` sets the directory that temperarily saves the factorization machine model.

`--token` sets the token required for solving QUBO by an Ising machine.

Default Options:

`--factor 8` means the factor number in the factorization machine is 8. Higher number can fit more complex interactions.

`--patience 300` means the training of factorization machine stops without loss reduction after 300 epoch.

`--target max` means to maximize the optimization target. For minimization, set `--target min` instead.

`--num 300` means to output 300 optimized molecules.

`--client amplify` sets the Ising machine as Amplify.

## Amplify Token

The token can be registered freely at https://amplify.fixstars.com/en/.
