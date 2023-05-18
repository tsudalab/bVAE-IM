from rdkit import Chem
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import random
import torch
import sascorer
import pickle
import os

from rdkit.Chem import QED
from rdkit import DataStructs
from rdkit.Chem import AllChem


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed_all(908530)

with open("zinc/train.txt", "r") as f:
    smiles = f.readlines()


gsk3_path = os.path.join(os.path.dirname(__file__), 'gsk3/gsk3.pkl')
jnk3_path = os.path.join(os.path.dirname(__file__), 'jnk3/jnk3.pkl')
with open(gsk3_path, "rb") as f:
    gsk3_model = pickle.load(f)
with open(jnk3_path, "rb") as f:
    jnk3_model = pickle.load(f)

def fingerprints_from_mol(mol):
    features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)
    return features.reshape(1, -1)


def get_prop(smiles):
    mask = []
    fps = []
    qed_score = []
    sa_score = []

    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        mask.append( int(mol is not None) )
        fp = fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
        fps.append(fp)

        if mol is not None:
            qed_score.append(QED.qed(mol))
            sa_score.append((10/sascorer.calculateScore(mol) - 1) / 9)
        else:
            qed_score.append(0)
            sa_score.append(10)
    fps = np.concatenate(fps, axis=0)

    gsk3_score = gsk3_model.predict_proba(fps)[:, 1]
    jnk3_score = jnk3_model.predict_proba(fps)[:, 1]

    gsk3_score = gsk3_score * np.array(mask)
    jnk3_score = jnk3_score * np.array(mask)

    return (gsk3_score, jnk3_score, qed_score, sa_score)


num_sample = 10000
samples = []
props = []

random.shuffle(smiles)

for i in tqdm(range(len(smiles)//1000+1)):

    smi = smiles[i*1000:(i+1)*1000]
    gsk3_score, jnk3_score, qed_score, sa_score = get_prop(smi)
    
    for g, j, q, sa, s in zip(gsk3_score, jnk3_score, qed_score, sa_score, smi):
        
        ans = pow(g*j*q*sa, 1/4)
        if g > 0.1 or j > 0.1:
            if ans < 0.5:
            # if smi not in samples:
                samples.append(s)
                props.append(ans)

    assert len(samples) == len(props)

np.save("opt/multi_train_smiles10k.npy", samples[:num_sample])
np.save("opt/multi_train_props10k.npy", props[:num_sample])