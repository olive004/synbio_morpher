

import numpy as np


def per_mol_to_per_molecules(jmol):
    """ Translate a value from the unit of per moles to per molecules.
    The number of M of mRNA in a cell was calculated using the average 
    number of mRNA in an E. coli cell (100 molecules) and the average volume of an E.
    coli cell (1.1e-15 L) to give ca. 1 molecule ~ 1.50958097 nM ~ 1.50958097e-9 M"""
    # J/mol to J/molecule
    # return np.divide(jmol, SCIENTIFIC['mole'])
    return np.divide(jmol, 1.50958097/np.power(10, 9))