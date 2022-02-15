from math import factorial
from copy import deepcopy
import numpy as np


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


def generate_mixed_binary(length, num_true, count):
    # TODO: This could be much better. Generate
    # sequences in a way that
    # 1. maximizes sequence orthogonality
    # 2. prioritises different neighbors
    #
    assert count <= length, f'Currently can only produce as many sequences as length of base' \
        'sequence. {count} sequences too many for length {length}.'
    interval = int(length / count)
    basic = [False] * length
    all_sequences = np.ones((count, length))
    for c in range(count):
        idxs = np.arange(c, length, interval)
        all_sequences[c, idxs] = 0

    return all_sequences


v = generate_mixed_binary(10, 3, 4)
print(v)
