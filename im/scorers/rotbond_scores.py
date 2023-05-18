from rdkit.Chem import rdMolDescriptors

def score_function(mol):
    score = rdMolDescriptors.CalcNumRotatableBonds(mol)
    return score