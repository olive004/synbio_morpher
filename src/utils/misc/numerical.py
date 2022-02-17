from math import factorial
from copy import deepcopy
import numpy as np


SCIENTIFIC = {
    # R = the gas constant = 8.314 J/mol·K
    # T = 298 K
    'RT': np.multiply(8.314, 298), # J/mol
    'mole': np.multiply(6.02214076, np.power(10, 23))
}


def zero_out_negs(npmatrix):
    return npmatrix.clip(min=0)


def nPr(n, r):
    return int(factorial(n)/factorial(n-r))


def binary_arpeggiator(sequence, count):
    length = len(sequence)
    interval = int(sequence / count)
    arpeggiation = np.arange(0, length, interval)

    for c in range(count):
        seq = np.zeros(length)
        seq[arpeggiation] = 1


def generate_mixed_binary(length, count):
    # TODO: This could be much better. Generate
    # sequences in a way that
    # 1. maximizes sequence orthogonality
    # 2. prioritises different neighbors
    assert count <= length, f'Currently can only produce as many sequences as length of base ' \
        'sequence. {count} sequences too many for length {length}.'
    interval = int(length / count)
    all_sequences = np.zeros((count, length))
    for c in range(count):
        idxs = np.arange(c, length, count)
        all_sequences[c, idxs] = 1

    return all_sequences

