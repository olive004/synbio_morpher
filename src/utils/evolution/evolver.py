from functools import partial
import logging
import os
import random
from typing import Tuple
import numpy as np
from bioreaction.model.data_containers import Species
from src.srv.io.loaders.misc import load_csv
from src.utils.evolution.mutation import get_mutation_type_mapping, Mutations
from src.utils.misc.type_handling import flatten_listlike
from src.utils.results.writer import DataWriter
from src.utils.misc.string_handling import add_outtype


from src.utils.circuit.agnostic_circuits.circuit_new import Circuit


class Evolver():

    def __init__(self, data_writer: DataWriter, mutation_type: str = 'random', sequence_type: str = None) -> None:
        self.data_writer = data_writer
        self.mutation_type = mutation_type  # Not implemented
        self.out_name = 'mutations'
        self.out_type = 'csv'
        self.sequence_type = sequence_type
        self.mutation_type_mapping = get_mutation_type_mapping(sequence_type)

    def is_mutation_possible(self, circuit: Circuit):
        if circuit.mutations_args['mutation_counts'] is None or circuit.mutations_args['mutation_nums_within_sequence'] is None:
            return False
        return True

    def mutate(self, circuit: Circuit, algorithm: str, write_to_subsystem=False):
        """ algorithm can be either random or all """
        if write_to_subsystem:
            self.data_writer.subdivide_writing(circuit.name)
        if self.is_mutation_possible(circuit):
            mutator = self.get_mutator(algorithm)
            circuit = mutator(circuit)
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

        def rand_mutator(circuit: Circuit, algorithm: str, positions_chosen=None):
            for i, specie in enumerate(circuit.model.species):
                circuit.mutations[specie.name] = {}
                sequence = specie.physical_data
                if not sequence:
                    continue
                for mutation_nums_within_sequence in circuit.mutations_args['mutation_nums_within_sequence']:
                    for mutation_counts in circuit.mutations_args['mutation_counts']:
                        for mutation_idx in range(mutation_counts):

                            positions = positions if positions_chosen is not None else mutation_sampler(
                                sequence, mutation_nums_within_sequence)
                            mutation_types, positions = self.sample_mutations(
                                sequence, positions, circuit.mutations_args['mutation_nums_per_position'])

                            mutation = self.make_mutations(
                                specie=specie, positions=positions, 
                                mutation_idx=mutation_idx,
                                mutation_types=mutation_types, algorithm=algorithm,
                                template_file=circuit.data.source)

                            circuit.mutations[specie.name][mutation.mutation_name] = mutation
            return circuit

        def all_mutator(circuit: Circuit, algorithm: str):
            mutation_nums_per_position = [len(
                self.mutation_type_mapping.keys()) - 1]
            for i, specie in enumerate(circuit.model.species):
                circuit.mutations[specie.name] = {}
                sequence = specie.physical_data
                for positions in ([i] for i in range(len(sequence))):
                    mutation_types, positions = self.sample_mutations(
                        sequence, positions, mutation_nums_per_position)
                    for i, (mt, p) in enumerate(zip(mutation_types, positions)):
                        mutation_idx = mutation_nums_per_position[0]*p + i
                        mutations = self.make_mutations(
                            species=specie, positions=[
                                p], mutation_idx=mutation_idx, mutation_types=[
                                mt], algorithm=algorithm, template_file=circuit.data.source)
                        circuit.mutations[specie.name][mutations.mutation_name] = mutations
            return circuit

        if algorithm == "random":
            return partial(rand_mutator, algorithm=algorithm)
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

    def make_mutations(self, specie: Species, positions: list,
                       mutation_idx: int, mutation_types: list,
                       algorithm: str, template_file: str):
        mutation_count = len(positions)
        mutations = Mutations(
            mutation_name=specie.name+'_' +
            f'm{mutation_count}-' + str(
                mutation_idx),
            template_species=specie,
            mutation_types=mutation_types,
            count=mutation_count,
            positions=positions,
            sequence_type=self.sequence_type,
            template_file=template_file,
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