

import numpy as np
from functools import partial
from synbio_morpher.utils.misc.numerical import make_symmetrical_matrix_from_sequence
import jax
import seaborn as sns
import matplotlib.pyplot as plt

RNA_EU = 0
RNA_EL = -1.5


def generate_energies(n_circuits: int, n_species: int, len_seq: int, p_null: float, symmetrical=True, type_energies: str = 'RNA') -> np.ndarray:
    """
    Generate a list of interaction energies for n_circuits circuits with n_species species each.
    The sequence length (len_seq) is the approximate length of the interacting components if they existed.
    For example, for RNA of length 20, the strongest interaction energy is different to RNAs of length 100.
    """
    fn = partial(transform_to_rna_energies, len_seq=len_seq,
                 p_null=p_null) if type_energies == 'RNA' else lambda x: x
    if symmetrical:
        n_interacting = int(np.sum(np.arange(n_species + 1)))
        energies = fn(np.random.rand(n_circuits, n_interacting))
        energies = np.array(jax.vmap(partial(
            make_symmetrical_matrix_from_sequence, side_length=n_species))(energies))
    else:
        energies = fn(np.random.rand(n_circuits, n_species, n_species))
    return energies


def pepper_noninteracting(energies: np.ndarray, p_null: float) -> np.ndarray:
    """ p_null is the percentage of non-interacting species """
    mask = np.random.choice(energies.size, size=int(
        energies.size * p_null), replace=False)
    energies.flat[mask] = 0
    return energies


def transform_to_rna_energies(rnd: np.ndarray, len_seq: int, p_null: float) -> np.ndarray:
    """
    Transform interaction energies to RNA interaction energies.
    """
    rnd = np.log(rnd + 5e-2)
    energies = np.interp(rnd, (rnd.min(), rnd.max()),
                         (RNA_EL*len_seq, RNA_EU*len_seq))
    energies = pepper_noninteracting(energies, p_null=p_null)
    return energies


def main():
    e = generate_energies(
        3000, 3, 20, 0.1, symmetrical=True, type_energies='RNA')
    sns.histplot(e[:, np.triu_indices(3)].flatten(), bins=50, element='step')
    plt.savefig('test.png')
