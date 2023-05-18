from rdkit.Chem import Descriptors

def score_function(mol):
    score = Descriptors.ExactMolWt(mol)
    return score