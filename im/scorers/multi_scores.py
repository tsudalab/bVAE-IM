import os
import pickle
from scorers import sascorer
import numpy as np

from rdkit.Chem import QED
from rdkit import DataStructs
from rdkit.Chem import AllChem

def fingerprints_from_mol(mol):
    features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)
    return features.reshape(1, -1)


def score_function(mol):
    gsk3_path = os.path.join(os.path.dirname(__file__), 'gsk3/gsk3.pkl')
    jnk3_path = os.path.join(os.path.dirname(__file__), 'jnk3/jnk3.pkl')
    with open(gsk3_path, "rb") as f:
        gsk3_model = pickle.load(f)
    with open(jnk3_path, "rb") as f:
        jnk3_model = pickle.load(f)

    fp = fingerprints_from_mol(mol)

    gsk3_score = gsk3_model.predict_proba(fp)[0, 1]
    jnk3_score = jnk3_model.predict_proba(fp)[0, 1]
    qed_score = QED.qed(mol)
    sa_score = sascorer.calculateScore(mol)
    sa_score = (10/sa_score - 1) / 9

    score = np.power(gsk3_score*jnk3_score*qed_score*sa_score, 1/4)
    print(score)

    return score
