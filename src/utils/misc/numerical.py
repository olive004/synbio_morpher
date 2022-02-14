from math import factorial


def zero_out_negs(npmatrix):
    return npmatrix.clip(min=0)


def nPr(n, r):
    return int(factorial(n)/factorial(n-r))
