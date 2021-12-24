import numpy as np


def square_matrix(num_nodes=3):
    dims = (num_nodes, num_nodes)
    return np.random.rand(*dims)
