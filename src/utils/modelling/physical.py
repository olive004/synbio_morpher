import numpy as np
from src.utils.misc.units import SCIENTIFIC


def F(E):
    """ Parameterisation of relative GFP fluorescence binding from 
    paper [Metabolic engineering of Escherichia coli using synthetic small regulatory RNAs](
    https://www.nature.com/articles/nbt.2461
    ). See the notebook `explanations/binding_energy_reparameterisation` for 
    more details.
    The binding energy is in units of kcal/mol """
    F = (1-0.01)/(1+np.exp(-(E/2 + 5))) + 0.01
    return F


def equilibrium_constant_reparameterisation(E, initial: np.array):
    """ The energy input E is $\Delta G$ in kcal/mol. 
    IMPORTANT: Using the mean initial quantity of all species, as this 
    equation was derived under the assumption that all unbound species 
    start with the same concentration and have the same interactions """
    # return 1/initial * (1/F(E) - 1)
    Fs = np.exp(-0.8 * (E + 10))
    return Fs/initial


def gibbs_K(E):
    """ In J/mol. dG = - RT ln(K) """
    K = np.exp(np.divide(-E, SCIENTIFIC['RT']))
    return K


def gibbs_K_cal(E):
    """ Translate interaction binding energy (kcal) to the
    equilibrium rate of binding.
    AG = - RT ln(K)
    AG = - RT ln(kb/kd)
    K = e^(- G / RT)
    """
    K = np.exp(np.divide(-E, SCIENTIFIC['RT_cal']))
    return K
