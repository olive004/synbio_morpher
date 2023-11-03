
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    


from typing import List
from copy import deepcopy
import numpy as np
from bioreaction.model.data_containers import Species
from synbio_morpher.utils.circuit.agnostic_circuits.circuit import Circuit
from synbio_morpher.utils.results.writer import Tabulated


EXCLUDED_NUCS = ['N']
mutation_type_mapping_RNA = {
    # Mutation idx from parent key to child key
    "A": {
        "C": 0,
        "G": 1,
        "U": 2
    },
    "C": {
        "A": 3,
        "G": 4,
        "U": 5
    },
    "G": {
        "A": 6,
        "C": 7,
        "U": 8
    },
    "U": {
        "A": 9,
        "C": 10,
        "G": 11
    },
    "N": {
        "A": 12,
        "C": 12,
        "G": 12,
        "U": 12
    }
}

mutation_type_mapping_DNA = {
    # Mutation idx from parent key to child key
    "A": {
        "C": 0,
        "G": 1,
        "T": 2
    },
    "C": {
        "A": 3,
        "G": 4,
        "T": 5
    },
    "G": {
        "A": 6,
        "C": 7,
        "T": 8
    },
    "T": {
        "A": 9,
        "C": 10,
        "G": 11
    }
}


def get_mutation_type_mapping(sequence_type):
    if sequence_type == 'RNA':
        return mutation_type_mapping_RNA
    elif sequence_type == 'DNA':
        return mutation_type_mapping_DNA
    else:
        raise ValueError(f'Unrecognised sequence type {sequence_type} provided - should be DNA or RNA.')


class Mutations(Tabulated):

    def __init__(self, mutation_name: str, template_species: Species,
                 template_name: str, template_seq: str,
                 template_file: str, positions: List[int], mutation_types: List[int],
                 count: int, sequence_type: str, algorithm: str) -> None:

        self.mutation_name = mutation_name
        self.template_species = template_species
        self.template_name = template_name # self.template_species.name
        self.template_seq = template_seq # self.template_species.physical_data
        self.mutation_types = mutation_types
        self.positions = positions
        self.count = count
        self.algorithm = algorithm
        self.sequence_type = sequence_type
        self.template_file = template_file
        super().__init__()

    def get_table_properties(self):
        exempt_keys = ['template_species']
        props = self.__dict__
        for k in exempt_keys:
            props.pop(k)
        return list(props.keys()), list(props.values())

    def get_sequence(self):
        seq = list(deepcopy(self.template_seq))
        for i, p in enumerate(self.positions):
            seq[int(p)] = self.reverse_mut_mapping(int(self.mutation_types[i]))
        return ''.join(seq)

    def reverse_mut_mapping(self, mut_encoding: int):
        mutation_type_mapping = get_mutation_type_mapping(self.sequence_type)
        for k, v in mutation_type_mapping.items():
            if mut_encoding in list(v.values()):
                for mut, enc in v.items():
                    if enc == mut_encoding:
                        return mut
        raise ValueError(
            f'Could not find mutation for mapping key {mut_encoding}.')


def implement_mutation(circuit: Circuit, mutation: Mutations):

    if mutation.template_name in [s.name for s in circuit.model.species]:
        sidx = [s.name for s in circuit.model.species].index(mutation.template_name)
        # circuit.model.species[sidx].name = mutation.mutation_name
        circuit.model.species[sidx].physical_data = mutation.get_sequence()
    else:
        raise KeyError(
            f'Could not find specie {mutation.template_name} in model for mutation {mutation.mutation_name}')

    circuit.qreactions.update_reactants(circuit.model)
    circuit.qreactions.update_reactions(circuit.model)
    return circuit


def reverse_mut_mapping(mut_encoding: int, sequence_type: str = 'RNA'):
    for k, v in get_mutation_type_mapping(sequence_type).items():
        if mut_encoding in list(v.values()):
            for mut, enc in v.items():
                if enc == mut_encoding:
                    return mut
    raise ValueError(
        f'Could not find mutation for mapping key {mut_encoding}.')


def apply_mutation_to_sequence(sequence: str, mutation_positions: List[int], mutation_types: List[str]) -> List[str]:
    sequence = np.array([*sequence])
    sequence[mutation_positions] = mutation_types
    return ''.join(sequence)

