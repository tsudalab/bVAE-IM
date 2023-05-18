from rdkit import Chem
import numpy as np

import random
import torch
import sascorer
import networkx as nx

from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed(908530)

with open("zinc/train.txt", "r") as f:
    smiles = f.readlines()
smiles = [s.strip("\n\r") for s in smiles]

# mols = [Chem.MolFromSmiles(s) for s in tqdm(smiles)]
# logp_scores = [Descriptors.MolLogP(m) for m in tqdm(mols)]
# sa_scores = [-sascorer.calculateScore(m) for m in tqdm(mols)]
# cycle_scores = [-cal_cycle_score(m) for m in tqdm(mols)]

# logP_mean = np.mean(logp_scores)
# logP_std = np.std(logp_scores)
# SA_mean = np.mean(sa_scores)
# SA_std = np.std(sa_scores)
# cycle_mean = np.mean(cycle_scores)
# cycle_std = np.std(cycle_scores)

def get_prop(mol):
    # mean and standard variance for penalized logp calculation
    logP_mean = 2.4577998006008803
    logP_std = 1.4334180962410332
    SA_mean = -3.0535377758058795
    SA_std = 0.834614385518919
    cycle_mean = -0.04829303989345987
    cycle_std = 0.28775759686956115

    def cal_cycle_score(mol):
        cycle_list = nx.cycle_basis(
            nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max([len(j) for j in cycle_list])
        if cycle_length <= 6:
            cycle_length = 0
        else:
            cycle_length = cycle_length - 6

        return cycle_length

    current_log_P_value = Descriptors.MolLogP(mol)
    current_SA_score = -sascorer.calculateScore(mol)
    current_cycle_score = -cal_cycle_score(mol)

    current_SA_score_normalized = (
        current_SA_score - SA_mean) / SA_std

    current_log_P_value_normalized = (
        current_log_P_value - logP_mean) / logP_std

    current_cycle_score_normalized = (
        current_cycle_score - cycle_mean) / cycle_std

    score = (current_SA_score_normalized +
                current_log_P_value_normalized +
                current_cycle_score_normalized)

    return score

num_sample = 10000
samples = []
props = []

while len(props) < num_sample:
    smi = random.choice(smiles)

    mol = Chem.MolFromSmiles(smi)
    score = get_prop(mol)

    if smi not in samples and -3 < score < 3:
        samples.append(smi)
        props.append(score)

np.save("opt/plogp_train_smiles10k.npy", samples)
np.save("opt/plogp_train_props10k.npy", props)