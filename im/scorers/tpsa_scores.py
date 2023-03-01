import networkx as nx
from rdkit.Chem import Descriptors
from scorers import sascorer

def score_function(mol):
    score = Descriptors.TPSA(mol)
    return score