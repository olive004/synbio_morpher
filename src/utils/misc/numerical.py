import logging
from math import factorial
from typing import Union
import numpy as np


SCIENTIFIC = {
    # R = the gas constant = 8.314 J/mol·K
    # T = 298 K
    'RT': np.multiply(8.314, 298),  # J/mol
    'mole': np.multiply(6.02214076, np.power(10, 23))
}


def zero_out_negs(npmatrix):
    return npmatrix.clip(min=0)


def nPr(n, r):
    return int(factorial(n)/factorial(n-r))


def binary_arpeggiator(sequence: str, count: int):
    length = len(sequence)
    interval = int(sequence / count)
    arpeggiation = np.arange(0, length, interval)

    for c in range(count):
        seq = np.zeros(length)
        seq[arpeggiation] = 1


def generate_mixed_binary(length: int, count: int, zeros_to_ones: bool = True):
    # TODO: This could be much better. Generate
    # sequences in a way that
    # 1. maximizes sequence orthogonality
    # 2. prioritises different neighbors
    assert count <= length, f'Currently can only produce as many sequences as length of base ' \
        'sequence. {count} sequences too many for length {length}.'
    interval = int(length / count)
    all_sequences = np.zeros((count, length)) \
        if zeros_to_ones else np.ones((count, length))
    for c in range(count):
        idxs = np.arange(c, length, count)
        all_sequences[c, idxs] = 1 if zeros_to_ones else 0

    return all_sequences


def np_delete_axes(array, rowcol: Union[list, int], axes: list):
    for axis in axes:
        array = np.delete(array, rowcol, axis=axis)
    return array


def make_dynamic_indexer(desired_axis_index_pairs: dict) -> tuple:
    """ For numpy array advanced indexing: if you know the desired index-axis pair, 
    but the axis is dynamic, use this function to create the appropriate indexing tuple """
    idxs = [0] * len(desired_axis_index_pairs)
    for axis, index in desired_axis_index_pairs.items():
        idxs[axis] = index
    return tuple(idxs)


def make_symmetrical_matrix_from_sequence(arr, side_length: int, total_dimensions: int = 2, sequence: str = 'triangular'):
    matrix = np.zeros(tuple([side_length]*total_dimensions))
    if sequence == 'triangular':
        for side in range(1, side_length+1):
            prev_triangle = triangular_sequence(side-1)
            curr_triangle_num = triangular_sequence(side)

            matrix[side-1, 0:side] = arr[prev_triangle:curr_triangle_num]
            matrix[0:side, side-1] = arr[prev_triangle:curr_triangle_num]
    else:
        raise NotImplementedError(
            f'Unknown numerical sequence type {sequence}')
    return matrix


def round_to_nearest(x, base):
    return base * round(x/base)


def square_matrix_rand(num_nodes: int = 3):
    dims = (num_nodes, num_nodes)
    return np.random.rand(*dims)


def transpose_arraylike(arraylike):
    if type(arraylike) == list:
        arraylike = np.array(arraylike)
    return np.transpose(arraylike)


def triangular_sequence(n: int) -> int:
    return int((n*(n+1))/2)
