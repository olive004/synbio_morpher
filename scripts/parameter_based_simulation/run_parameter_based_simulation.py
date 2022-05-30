import numpy as np

from src.utils.system_definition.agnostic_system.modelling import Deterministic


def main(config=None, writer=None):

    interaction_min = 0
    interaction_max = 1
    interaction_step_size = 0.05
    interaction_array = np.arange(
        interaction_min, interaction_max, interaction_step_size)
    num_species = 3

    matrix_dimensions = np.power(len(interaction_array), num_species)

    Deterministic()
