from copy import deepcopy
from functools import partial
import logging
import os
import random
import sys
from typing import Iterator, Tuple
import pandas as pd
import numpy as np
from src.srv.io.loaders.misc import load_csv
from src.utils.misc.type_handling import flatten_listlike
from src.utils.results.writer import DataWriter, Tabulated
from src.utils.misc.string_handling import add_outtype, prettify_logging_info


from src.utils.circuit.agnostic_circuits.base_circuit import BaseSpecies, BaseCircuit


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
        logging.warning(
            f'Unrecognised sequence type {sequence_type} provided - should be DNA or RNA.')
        sys.exit()


class Mutations(Tabulated):

    def __init__(self, mutation_name, template_name, template_file, template_seq,
                 positions, mutation_types, count: int, sequence_type: str, algorithm: str) -> None:
        self.mutation_name = mutation_name
        self.template_name = template_name
        self.template_seq = template_seq
        self.mutation_types = mutation_types
        self.positions = positions
        self.count = count
        self.algorithm = algorithm
        self.sequence_type = sequence_type
        self.template_file = template_file

        super().__init__()

    def get_table_properties(self):
        return list(self.__dict__.keys()), list(self.__dict__.values())

    def get_sequence(self):
        seq = list(deepcopy(self.template_seq))
        for i, p in enumerate(self.positions):
            seq[p] = self.reverse_mut_mapping(self.mutation_types[i])
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


class Evolver():

    def __init__(self, data_writer: DataWriter, mutation_type: str = 'random', sequence_type: str = None) -> None:
        self.data_writer = data_writer
        self.mutation_type = mutation_type  # Not implemented
        self.out_name = 'mutations'
        self.out_type = 'csv'
        self.sequence_type = sequence_type
        self.mutation_type_mapping = get_mutation_type_mapping(sequence_type)

    def is_mutation_possible(self, system: BaseCircuit):
        if system.species.mutation_counts is None or system.species.mutation_nums_within_sequence is None:
            return False
        return True

    def mutate(self, circuit: BaseCircuit, algorithm: str, write_to_subsystem=False):
        """ algorithm can be either random or all """
        if write_to_subsystem:
            self.data_writer.subdivide_writing(circuit.name)
        if self.is_mutation_possible(circuit):
            mutator = self.get_mutator(algorithm)
            circuit.species = mutator(circuit.species)
        else:
            logging.info('No mutation settings found, did not mutate.')
        return circuit

    def get_mutator(self, algorithm):

        def mutation_sampler(sequence, num_mutations):
            if len(sequence) < num_mutations:
                logging.warning(
                    f'For sequences of length {len(sequence)}, can not mutate {num_mutations} times.')
            positions = random.sample(range(len(sequence)), num_mutations)
            return positions

        def basic_mutator(species: BaseSpecies, position_generator,
                          mutation_nums_within_sequence,
                          mutation_nums_per_position: list,
                          sample_idx: int = None,
                          mutation_idx: int = None,
                          positions: list = None) -> Mutations:
            
            return mutations

        def rand_mutator(species: BaseSpecies, sample_mutator_func, algorithm):
            for sample_idx, sample in enumerate(species.data.sample_names):
                species.mutations[sample] = {}
                for mutation_nums_within_sequence in species.mutation_nums_within_sequence:
                    for mutation_idx in range(species.mutation_counts[sample_idx]):
                        # mutation = sample_mutator_func(
                        #     species=species, sample_idx=sample_idx, mutation_idx=mutation_idx,
                        #     mutation_nums_within_sequence=mutation_nums_within_sequence,
                        #     mutation_nums_per_position=species.mutation_nums_per_position,
                        #     algorithm=algorithm)

                        sequence = species.data.get_data_by_idx(sample_idx)
                        positions = positions if positions is not None else mutation_sampler(
                            sequence, mutation_nums_within_sequence)
                        mutation_types, positions = self.sample_mutations(
                            sequence, positions, species.mutation_nums_per_position)

                        mutation = self.make_mutations(
                            species=species, positions=positions, sample_idx=sample_idx,
                            mutation_idx=mutation_idx, sequence=sequence,
                            mutation_types=mutation_types)

                        species.mutations[sample][mutation.mutation_name] = mutation
            return species

        def all_mutator(species: BaseSpecies, algorithm: str):
            mutation_nums_per_position = [len(
                self.mutation_type_mapping.keys()) - 1]
            for sample_idx, sample in enumerate(species.data.sample_names):
                species.mutations[sample] = {}
                sequence = species.data.get_data_by_idx(sample_idx)
                for positions in ([i] for i in range(len(sequence))):
                    mutation_types, positions = self.sample_mutations(
                        sequence, positions, mutation_nums_per_position)
                    for i, (mt, p) in enumerate(zip(mutation_types, positions)):
                        mutation_idx = mutation_nums_per_position[0]*p + i
                        mutations = self.make_mutations(
                            species=species, sample_idx=sample_idx, positions=[
                                p], mutation_idx=mutation_idx, sequence=sequence, mutation_types=[
                                mt], algorithm=algorithm)
                        species.mutations[sample][mutations.mutation_name] = mutations
            return species

        if algorithm == "random":
            return partial(rand_mutator, algorithm=algorithm,
                           sample_mutator_func=partial(basic_mutator,
                                                       position_generator=mutation_sampler))
        elif algorithm == "all":
            return partial(all_mutator, algorithm=algorithm)
        else:
            return ValueError(f'Unrecognised mutation algorithm choice "{algorithm}"')

    def sample_mutations(self, sequence: str, positions: list, mutation_nums_per_position: list) -> Tuple[list, list]:
        mutation_types = {}
        new_positions = []
        mutation_nums_per_position = mutation_nums_per_position * \
            len(positions) if len(
                mutation_nums_per_position) == 1 else mutation_nums_per_position
        for p, n in zip(positions, mutation_nums_per_position):
            possible_transitions = self.mutation_type_mapping[sequence[p]]
            if n > len(possible_transitions):
                logging.warning(
                    f'Cannot pick {n} when there are only {len(possible_transitions)} choices')
            mutation_types[p] = list(np.random.choice(
                list(possible_transitions.values()), size=n, replace=False))
            new_positions.append([p] * n)

        return flatten_listlike(mutation_types.values()), flatten_listlike(new_positions)

    def make_mutations(self, species: BaseSpecies, positions: list, sample_idx: int,
                       mutation_idx: int, sequence: str, mutation_types: list,
                       algorithm: str):
        mutation_count = len(positions)
        mutations = Mutations(
            mutation_name=species.data.sample_names[sample_idx]+'_' +
            f'm{mutation_count}-' + str(
                mutation_idx),
            template_name=species.data.sample_names[sample_idx],
            template_seq=sequence,
            mutation_types=mutation_types,
            count=mutation_count,
            positions=positions,
            sequence_type=self.sequence_type,
            template_file=species.data.source,
            algorithm=algorithm
        )
        self.write_mutations(mutations)
        return mutations

    def write_mutations(self, mutations: Mutations, overwrite=False):
        self.data_writer.output(
            out_type=self.out_type, out_name=self.out_name, data=mutations.as_table(), overwrite=overwrite)

    def load_mutations(self):
        filename = os.path.join(
            self.data_writer.write_dir, add_outtype(self.out_name, self.out_type))
        return load_csv(filename, load_as='dict')
