from copy import deepcopy
from functools import partial
import os
import random
import pandas as pd
import numpy as np
from src.utils.data.manage.writer import DataWriter, Tabulated
from src.utils.misc.string_handling import add_outtype


from src.utils.system_definition.agnostic_system.base_system import BaseSpecies, BaseSystem


mapping = {
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
    },
    "U": {
        "A": 12,
        "C": 13,
        "G": 14
    }
}


class Mutations(Tabulated):

    def __init__(self, mutation_name, template_name, template_file, template_seq,
                 positions, mutation_types, algorithm='random') -> None:
        self.mutation_name = mutation_name
        self.template_name = template_name
        self.template_file = template_file
        self.template_seq = template_seq
        self.mutation_types = mutation_types
        self.positions = positions
        self.count = len(positions)
        self.algorithm = algorithm

        super().__init__()

    def get_props_as_split_dict(self):
        return list(self.__dict__.keys()), list(self.__dict__.values())

    def get_sequence(self):
        seq = deepcopy(self.template_seq)
        for i, p in enumerate(self.positions):
            seq[p] = self.reverse_mut_mapping(mapping[i])
        return self.template_seq

    def reverse_mut_mapping(self, mut_encoding: int):
        for k, v in mapping.items():
            if mut_encoding in list(v.values()):
                return k
        raise ValueError(
            f'Could not find mutation mapping key {mut_encoding}.')


class Evolver():

    def __init__(self, data_writer: DataWriter) -> None:
        self.data_writer = data_writer
        self.out_name = 'mutations'
        self.out_type = 'csv'

    def mutate(self, system: BaseSystem, algorithm="random"):
        mutator = self.get_mutator(algorithm)
        system.species = mutator(system.species)
        return system

    def get_mutator(self, algorithm):

        def random_mutator(sequence, num_mutations):
            positions = list(np.random.randint(
                0, len(sequence), size=num_mutations))
            return positions

        def basic_mutator(species: BaseSpecies, position_generator, sample_idx: int = None,
                          mutation_idx=None):
            sequence = species.data.get_data_by_idx(sample_idx)
            positions = position_generator(
                sequence, species.mutation_nums[sample_idx])

            mutations = Mutations(
                mutation_name=species.data.sample_names[sample_idx]+'_'+str(mutation_idx),
                template_name=species.data.sample_names[sample_idx],
                template_file=species.data.source,
                template_seq=sequence,
                mutation_types=self.sample_mutations(sequence, positions),
                positions=positions
            )
            self.write_mutations(mutations)
            return mutations

        def full_mutator(species: BaseSpecies, sample_mutator_func):
            for sample_idx, sample in enumerate(species.data.sample_names):
                species.mutations[sample] = {}
                for c in range(species.mutation_counts[sample_idx]):
                    species.mutations[sample][c] = sample_mutator_func(
                        species, sample_idx=sample_idx, mutation_idx=c)
            return species

        if algorithm == "random":
            return partial(full_mutator, sample_mutator_func=partial(basic_mutator, position_generator=random_mutator)
                           )
        else:
            return ValueError(f'Unrecognised mutation algorithm choice "{algorithm}"')

    def sample_mutations(self, sequence, positions):
        mutation_types = []
        for p in positions:
            possible_transitions = mapping[sequence[p]]
            mutation_types.append(random.choice(
                list(possible_transitions.values())))
        return mutation_types

    def write_mutations(self, mutations: Mutations, overwrite=False):
        self.data_writer.output(
            out_type=self.out_type, out_name=self.out_name, data=mutations.as_table(), overwrite=overwrite)

    def load_mutations(self):
        filename = os.path.join(
            self.data_writer.write_dir, add_outtype(self.out_name, self.out_type))
        return pd.read_csv(filename).to_dict()
