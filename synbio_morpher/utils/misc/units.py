
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    


import numpy as np


SCIENTIFIC = {
    # R = the gas constant = 8.314 J/mol·K
    # T = 298 K
    'RT': np.multiply(8.314, 310),  # J/mol
    'RT_cal': np.multiply(1.987, 310),
    'mole': np.multiply(6.02214076, np.power(10, 23)),
    'cal_to_J_factor': 4.184
}


def per_mol_to_per_molecule(per_mol):
    """ Translate a value from the unit of per moles to per molecules.
    The number of M of mRNA in a cell was calculated using the average 
    number of mRNA in an E. coli cell (100 molecules) and the average volume of an E.
    coli cell (1.1e-15 L) to give ca. 1 molecule ~ 1.50958097 nM ~ 1.50958097e-9 M"""
    # 1/mol to 1/molecule
    # return np.divide(jmol, SCIENTIFIC['mole'])
    return np.multiply(per_mol, 1.50958097/np.power(10, 9))


def cal_to_J(E_cal):
    """ Convert calories to joules """
    return E_cal * SCIENTIFIC['cal_to_J_factor']


def j_to_cal(E_j):
    return E_j / SCIENTIFIC['cal_to_J_factor']


def unkilo(kilo):
    """ A unit in kilos is turned into its equivalent un-kilos """
    return kilo * 1000
