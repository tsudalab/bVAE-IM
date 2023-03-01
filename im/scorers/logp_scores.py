import networkx as nx
from rdkit.Chem import rdmolops, Descriptors
from scorers import sascorer

def score_function(mol):
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