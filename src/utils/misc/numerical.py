import functools
import logging
from math import factorial
from typing import Union, List
import numpy as np
import re
import pandas as pd


SCIENTIFIC = {
    # R = the gas constant = 8.314 J/mol·K
    # T = 298 K
    'RT': np.multiply(8.314, 310),  # J/mol
    'mole': np.multiply(6.02214076, np.power(10, 23))
}

NUMERICAL = {
    'infinity': np.multiply(10, 10),
    'nan': 0
}


def cast_astype(list_like, dtypes):
    if type(dtypes) == list:
        dtype = dtypes[0]
        if len(dtypes) == 1:
            dtypes = dtypes[0]
        else:
            dtypes = dtypes[1:]
        list_like = cast_astype(list_like, dtypes)
    else:
        dtype = dtypes
    if type(list_like) == list:
        recast = map(dtype, list_like)
    elif type(list_like) == pd.Series:
        def replace_string_list(e):
            if type(e) == str and re.search("\[.*\]", e):
                # return [dtype(ll) for ll in e[1:-1].split(',')]
                return dtype(e[1:-1])
            elif type(e) == list:
                # return [dtype(ee) for ee in e]
                return dtype(e[0])
            else:
                return dtype(e)
        recast = list_like.apply(replace_string_list)
        # recast = list_like.astype(dtype)
    else:
        raise TypeError(
            f'Type {type(list_like)} cannot be converted to {dtype}')
    return recast


def binary_arpeggiator(sequence: str, count: int):
    length = len(sequence)
    interval = int(sequence / count)
    arpeggiation = np.arange(0, length, interval)

    for c in range(count):
        seq = np.zeros(length)
        seq[arpeggiation] = 1


def calculate_num_decimals(floatlike):
    return int(np.log10(float(str(floatlike)[::-1]))) + 1


def expand_matrix_triangle_idx(flat_triangle_idx):
    """ Computes the indices of a triangle, or the lower half
    of a symmetrical square matrix. Assuming that 1D index 
    traverses columns then rows.
    For example, a 1D flat input of 5 would index into the 
    3rd row and 2nd column, returning the 0-based indices (2, 1) """

    # Reverse of the triangle_sequence formula
    n = (-1 + np.sqrt(1 - 4 * 1 * (-2*flat_triangle_idx))) / 2
    row = np.floor(n)
    col = flat_triangle_idx - triangular_sequence(np.floor(n))

    return (int(row), int(col))


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


def make_dynamic_indexer(desired_axis_index_pairs: dict) -> tuple:
    """ For numpy array advanced indexing: if you know the desired index-axis pair, 
    but the axis is dynamic, use this function to create the appropriate indexing tuple """
    idxs = [0] * len(desired_axis_index_pairs)
    for axis, index in desired_axis_index_pairs.items():
        idxs[axis] = index
    return tuple(idxs)


def init_matrices(self, uniform_vals, ndims=2, init_type="rand") -> List[np.array]:
    matrices = (self.init_matrix(ndims, init_type, val)
                for val in uniform_vals)
    return tuple(matrices)


def init_matrix(self, ndims=2, init_type="rand", uniform_val=1) -> np.array:
    matrix_size = np.random.randint(5) if self.data is None \
        else self.data.size
    if ndims > 1:
        matrix_shape = tuple([matrix_size]*ndims)
    else:
        matrix_shape = (matrix_size, 1)

    if init_type == "rand":
        return square_matrix_rand(matrix_size)
    elif init_type == "randint":
        return np.random.randint(10, 1000, matrix_shape).astype(np.float64)
    elif init_type == "uniform":
        return np.ones(matrix_shape) * uniform_val
    elif init_type == "zeros":
        return np.zeros(matrix_shape)
    raise ValueError(f"Matrix init type {init_type} not recognised.")


def invert_onehot(onehot):
    return (onehot == 0).astype(onehot.dtype)


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


def np_delete_axes(array, rowcol: Union[list, int], axes: list):
    for axis in axes:
        array = np.delete(array, rowcol, axis=axis)
    return array


def nPr(n, r):
    return int(factorial(n)/factorial(n-r))


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


def zero_out_negs(npmatrix):
    return npmatrix.clip(min=0)


pass
