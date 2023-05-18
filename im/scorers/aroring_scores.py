from rdkit.Chem import rdMolDescriptors

def score_function(mol):
    score = rdMolDescriptors.CalcNumAromaticRings(mol)
    return score