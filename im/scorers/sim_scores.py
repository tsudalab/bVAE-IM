from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

def score_function(mol):
    target_smiles = "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1"
    target_mol = Chem.MolFromSmiles(target_smiles)
    target_fp = AllChem.GetMorganFingerprint(target_mol, radius=2)

    fp = AllChem.GetMorganFingerprint(mol, radius=2)
    score = DataStructs.TanimotoSimilarity(target_fp, fp)
    return score

