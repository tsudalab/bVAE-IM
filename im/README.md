# Optimization on the Binary  Latent Space

## Optimization

Here gives an example for optimizing penalized LogP with the factorization machine based on 10k labeled data.

```
python bVAE-IM.py -y config/amplify_logp_fm.yaml
```

Other experiments can be duplicated by replacing the config file with others.

Modify the random seed in `.yaml` to get different optimization results.

In our experiments, the random seed is set as 1-5 respectively for the 5 runs.

## Amplify Token

The token can be registered freely at https://amplify.fixstars.com/en/.
