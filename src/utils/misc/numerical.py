from math import factorial
from copy import deepcopy
import numpy as np


def zero_out_negs(npmatrix):
    return npmatrix.clip(min=0)


def nPr(n, r):
    return int(factorial(n)/factorial(n-r))


def generate_mixed_binary(length, num_true, count):
    basic = [False] * length
    all_sequences = np.ones((count, length))
    matrix[idxs] = 0
    for c in range(count):
        idxs = create_new_pattern(c, count, length, num_true)
        all_sequences[c, idxs] 
    
    return interval


v = generate_mixed_binary(10, 3, 3)
print(v)