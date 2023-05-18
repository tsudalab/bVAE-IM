from rdkit import Chem
import numpy as np

import random
import torch
from tqdm import tqdm

from rdkit.Chem import rdMolDescriptors


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed(908530)

with open("zinc/train.txt", "r") as f:
    smiles = f.readlines()
smiles = [s.strip("\n\r") for s in smiles]

def get_prop(mol):
    return rdMolDescriptors.CalcNumRotatableBonds(mol)

mols = [Chem.MolFromSmiles(s) for s in tqdm(smiles)]

num_sample = 10000
samples = []
props = []

while len(props) < num_sample:
    smi = random.choice(smiles)

    mol = Chem.MolFromSmiles(smi)
    score = get_prop(mol)

    if smi not in samples and 0 <= score <= 5:
        samples.append(smi)
        props.append(score)

np.save("opt/rotbond_train_smiles10k.npy", samples)
np.save("opt/rotbond_train_props10k.npy", props)