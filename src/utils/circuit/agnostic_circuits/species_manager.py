# from typing import List
# import numpy as np
# import pandas as pd
# from copy import deepcopy
# import logging


# from src.utils.misc.numerical import square_matrix_rand
# import bioreaction


# class Species(bioreaction.model.data_containers.Species):

#     species_axis = 0
#     time_axis = 1

#     def __init__(self, name, identity=None) -> None:
#         super(self).__init__(name)
#         self.identity = identity


# class QReactionsManager():

#     def init_rates(self, config):
#         self.are_interactions_loaded = False
#         self.interactions, self.interaction_units = self.make_interactions(
#             config)
#         self.degradation_rates = self.init_matrix(ndims=1, init_type="uniform")
#         self.creation_rates = self.init_matrix(ndims=1, init_type="uniform")

#     def init_state(self):
#         self.copynumbers = None
#         self.current_copynumbers = None
#         self.steady_state_copynums = self.init_matrix(
#             ndims=1, init_type="zeros")


# class SpeciesManager():

#     species_axis = 0
#     time_axis = 1

#     def __init__(self, config: dict = None) -> None:
#         pass

#     def process_identities(self, identities: dict):
#         """ Make sure identities are indices of the sample names list """
#         for category, identity in identities.items():
#             if identity in self.data.sample_names:
#                 identities[category] = self.data.sample_names.index(identity)
#         return identities

#     def init_matrices(self, uniform_vals, ndims=2, init_type="rand") -> List[np.array]:
#         matrices = (self.init_matrix(ndims, init_type, val)
#                     for val in uniform_vals)
#         return tuple(matrices)

#     def init_matrix(self, ndims=2, init_type="rand", uniform_val=1) -> np.array:
#         matrix_size = np.random.randint(5) if self.data is None \
#             else self.data.size
#         if ndims > 1:
#             matrix_shape = tuple([matrix_size]*ndims)
#         else:
#             matrix_shape = (matrix_size, 1)

#         if init_type == "rand":
#             return square_matrix_rand(matrix_size)
#         elif init_type == "randint":
#             return np.random.randint(10, 1000, matrix_shape).astype(np.float64)
#         elif init_type == "uniform":
#             return np.ones(matrix_shape) * uniform_val
#         elif init_type == "zeros":
#             return np.zeros(matrix_shape)
#         raise ValueError(f"Matrix init type {init_type} not recognised.")

#     def init_mutations(self, config: dict):
#         self.mutations = {}
#         self.mutation_nums_within_sequence = config.get(
#             "mutations_args", {}).get("mutation_nums_within_sequence")
#         self.mutation_nums_per_position = config.get(
#             "mutations_args", {}).get("mutation_nums_per_position")
#         # Counts: mutated iterations of a sequence
#         self.mutation_counts = config.get(
#             "mutations_args", {}).get("mutation_counts")
#         self.process_mutations()

#     def make_interactions(self, config):
#         cfg_interactions = config.get("interactions")
#         if cfg_interactions:
#             if cfg_interactions.get("interactions_path", None):
#                 matrix, interaction_units = InteractionMatrix().load(
#                     cfg_interactions.get("interactions_path"))
#                 self.are_interactions_loaded = True
#             elif cfg_interactions.get("interactions_matrix", None) is not None:
#                 matrix, interaction_units = InteractionMatrix(
#                     matrix=cfg_interactions.get("interactions_matrix")).matrix, \
#                     cfg_interactions.get("interactions_units", '')
#                 self.are_interactions_loaded = True
#             else:
#                 logging.warning(f'No interactions could be loaded.')
#         else:
#             matrix, interaction_units = self.init_matrix(
#                 ndims=2, init_type="zeros"), ''
#         return (matrix, interaction_units)

#     def process_mutations(self):
#         self.mutation_counts = extend_int_to_list(
#             self.mutation_counts, self.size)
#         self.mutation_nums_within_sequence = extend_int_to_list(
#             self.mutation_nums_within_sequence, self.size)

#     def mutate(self, mutation):

#         if mutation.template_name in self.data.data.keys():
#             self.data.data[mutation.mutation_name] = mutation.get_sequence()
#             del self.data.data[mutation.template_name]
#         else:
#             raise KeyError(
#                 f'Could not find specie {mutation.template_name} in data for mutation {mutation.mutation_name}')

#         if mutation.template_name in self.data.identities.values():
#             for k, v in self.data.identities.items():
#                 if v == mutation.template_name:
#                     self.data.identities[k] = mutation.mutation_name

#         self.data.sample_names = self.data.make_sample_names()

#     def interactions_to_df(self, interactions: np.ndarray):
#         interactions_df = pd.DataFrame.from_dict(
#             self.interactions_to_dict(interactions))
#         return interactions_df

#     def interactions_to_dict(self, interactions: np.ndarray):
#         interactions_dict = {}
#         for i, sample in enumerate(self.data.sample_names):
#             interactions_dict[sample] = {s: interactions[i, j]
#                                          for j, s in enumerate(self.data.sample_names)}
#         return interactions_dict

#     def save_all_values(self):
#         return {prop: deepcopy(val) for prop, val in self.__dict__.items()}

#     def reset_to_initial_state(self):
#         resetable_attrs = ['interactions', 'copynumbers', 'loaded_interactions',
#                            'current_copynumbers', 'steady_state_copynums']
#         for k, v in self.initial_values.items():
#             if k in resetable_attrs:
#                 self.__setattr__(k, deepcopy(v))
