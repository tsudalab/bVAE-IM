import networkx as nx
from rdkit.Chem import QED
from scorers import sascorer

def score_function(mol):
    score = QED.qed(mol)
    return score