import logging
from random import random
import numpy as np


from src.utils.system_definition.agnostic_system.base_system import BaseSpecies


class Mutations():

    def __init__(self, mutation_name, template_file, template_species,
                 positions, mutation_types, algorithm='random') -> None:
        self.mutation_name = mutation_name
        self.template_file = template_file
        self.template_species = template_species
        self.mutation_types = mutation_types
        self.positions = positions
        self.count = len(positions)
        self.algorithm = algorithm


class Mutator():

    def __init__(self) -> None:
        self.mutation_level = 0
        self.curve


class Evolver():

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

    def __init__(self) -> None:
        self.mutation_num = 0
        
    def mutate(self, data, algorithm, **specs):
        mutator = self.get_mutator()

        return mutator(data)

    def get_mutator(self, algorithm):

        def random_mutator(species: BaseSpecies, species_idx: int):
            sequence = species.data.get_data_by_idx(species_idx)
            positions = np.random.randint(0, sequence, size=self.mutation_num)
            Mutations(
                mutation_name=species.data.sample_names,
                template_file=species.data.source,
                template_species=sequence,
                mutation_types=self.sample_mutations(sequence, positions),
                positions=positions
            )
            self.mutation_num
        
        if algorithm == "random":
            return random_mutator()

    def sample_mutations(self, sequence, positions):
        mutation_types = []
        for p in positions:
            possible_transitions = self.mapping[sequence[p]]
            logging.info(possible_transitions)
            mutation_types.append(random.choice(list(possible_transitions.keys())))
        return mutation_types