

import numpy as np
from functools import partial


RNA_EU = 0
RNA_EL = -1.5


def transform_to_rna_energies(rnd, len_seq):
    """
    Transform interaction energies to RNA interaction energies.
    """
    rnd = np.log(rnd)
    return np.interp(rnd, (rnd.min(), rnd.max()), (RNA_EL*len_seq, RNA_EU*len_seq))


def generate_energies(n_circuits, n_species, len_seq, symmetrical=True, type_energies='RNA'):
    """
    Generate a list of interaction energies for n_circuits circuits with n_species species each.
    The sequence length (len_seq) is the approximate length of the interacting components if they existed.
    For example, for RNA of length 20, the strongest interaction energy is different to RNAs of length 100.
    """
    # n_interactions = n_species if not symmetrical else np.sum(np.arange(n_species+1))
    fn = partial(transform_to_rna_energies, len_seq=len_seq) if type_energies == 'RNA' else lambda x: x
    energies = fn(np.random.rand(n_circuits, n_species))
    if symmetrical:
        energies = np.triu(energies)
        energies += energies.T
    return energies


generate_energies(2,3, 20)