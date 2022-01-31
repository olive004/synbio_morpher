import numpy as np


def zero_out_negs(npmatrix):
    return npmatrix.clip(min=0)