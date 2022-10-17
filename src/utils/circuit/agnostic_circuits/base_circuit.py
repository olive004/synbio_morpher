from copy import deepcopy
from typing import List
import numpy as np
import pandas as pd
import networkx as nx
import logging

from src.srv.parameter_prediction.interactions import InteractionMatrix
from src.utils.results.results import ResultCollector
from src.utils.misc.type_handling import extend_int_to_list


FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
FORMAT = "%(filename)s:%(funcName)s():%(lineno)i: %(message)s %(levelname)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseSpecies():

    species_axis = 0
    time_axis = 1

    def __init__(self, config_args: dict) -> None:

        # Probability distribution for each interaction component?
        # Can also use different types of moments of prob distribution
        # for each

        self.data = config_args.get("data")  # Data common class
        self.identities = self.process_identities(config_args.get("identities"))

        self.loaded_interactions = False
        self.interactions, self.interaction_units = self.make_interactions(
            config_args)
        self.degradation_rates = self.init_matrix(ndims=1, init_type="uniform")
        self.creation_rates = self.init_matrix(ndims=1, init_type="uniform")

        self.copynumbers = None
        self.current_copynumbers = None
        self.steady_state_copynums = self.init_matrix(
            ndims=1, init_type="zeros")

        self.mutations = {}
        # Nums: mutations within a sequence
        self.mutation_nums_within_sequence = config_args.get(
            "mutations", {}).get("mutation_nums_within_sequence")
        logging.info(self.mutation_nums_within_sequence)
        # Counts: mutated iterations of a sequence
        self.mutation_counts = config_args.get(
            "mutations", {}).get("mutation_counts")
        logging.info(self.mutation_counts)

        self.process_mutations()
        self.initial_values = self.save_initial_values()

    def process_identities(self, identities: dict):
        """ Make sure identities are indices of the sample names list """
        for category, identity in identities.items():
            if identity in self.data.sample_names:
                identities[category] = self.data.sample_names.index(identity)
        return identities

    def make_interactions(self, config_args):
        cfg_interactions = config_args.get("interactions")
        if cfg_interactions:
            if cfg_interactions.get("interactions_path", None):
                self.loaded_interactions = True
                matrix, interaction_units = InteractionMatrix().load(
                    cfg_interactions.get("interactions_path"))
            elif cfg_interactions.get("interactions_matrix", None) is not None:
                self.loaded_interactions = True
                matrix, interaction_units = InteractionMatrix(
                    matrix=cfg_interactions.get("interactions_matrix")).matrix, \
                    cfg_interactions.get("interactions_units", '')
        else:
            matrix, interaction_units = self.init_matrix(
                ndims=2, init_type="zeros"), ''
        return (matrix, interaction_units)

    def init_matrices(self, uniform_vals, ndims=2, init_type="rand") -> List[np.array]:
        matrices = (self.init_matrix(ndims, init_type, val)
                    for val in uniform_vals)
        return tuple(matrices)

    def init_matrix(self, ndims=2, init_type="rand", uniform_val=1) -> np.array:
        matrix_size = np.random.randint(5) if self.data is None \
            else self.data.size
        if ndims > 1:
            matrix_shape = tuple([matrix_size]*ndims)
        else:
            matrix_shape = (matrix_size, 1)

        if init_type == "rand":
            return InteractionMatrix(num_nodes=matrix_size).matrix
        elif init_type == "randint":
            return np.random.randint(10, 1000, matrix_shape).astype(np.float64)
        elif init_type == "uniform":
            return np.ones(matrix_shape) * uniform_val
        elif init_type == "zeros":
            return np.zeros(matrix_shape)
        raise ValueError(f"Matrix init type {init_type} not recognised.")

    def process_mutations(self):
        self.mutation_counts = extend_int_to_list(
            self.mutation_counts, self.size)
        self.mutation_nums_within_sequence = extend_int_to_list(self.mutation_nums_within_sequence, self.size)
        if self.mutation_counts is None or self.mutation_nums_within_sequence is None:
            logging.warning(f'Processing mutations may have gone wrong.')

    def mutate(self, mutation):

        if mutation.template_name in self.data.data.keys():
            self.data.data[mutation.mutation_name] = mutation.get_sequence()
            del self.data.data[mutation.template_name]
        else:
            raise KeyError(
                f'Could not find specie {mutation.template_name} in data for mutation {mutation.mutation_name}')

        if mutation.template_name in self.data.identities.values():
            for k, v in self.data.identities.items():
                if v == mutation.template_name:
                    self.data.identities[k] = mutation.mutation_name

        self.data.sample_names = self.data.make_sample_names()

    @property
    def interactions(self):
        return self._interactions

    @interactions.setter
    def interactions(self, new_interactions):
        if type(new_interactions) == np.ndarray:
            self._interactions = new_interactions
        else:
            raise ValueError('Cannot set interactions to' +
                             f' type {type(new_interactions)}.')

    def interactions_to_df(self, interactions):
        interactions_df = pd.DataFrame.from_dict(self.interactions_to_dict(interactions))
        return interactions_df

    def interactions_to_dict(self, interactions):
        interactions_dict = {}
        for i, sample in enumerate(self.data.sample_names):
            interactions_dict[sample] = {s: interactions[i, j]
                                         for j, s in enumerate(self.data.sample_names)}
        return interactions_dict

    def save_initial_values(self):
        return {prop: deepcopy(val) for prop, val in self.__dict__.items()}

    def reset_to_initial_state(self):
        resetable_attrs = ['interactions', 'copynumbers', 'loaded_interactions',
                           'current_copynumbers', 'steady_state_copynums']
        for k, v in self.initial_values.items():
            if k in resetable_attrs:
                self.__setattr__(k, deepcopy(v))

    @property
    def current_copynumbers(self):
        if self.copynumbers is not None:
            self._current_copynumbers = self.copynumbers[:, -1]
        else:
            self._current_copynumbers = self._copynumbers
        return self._current_copynumbers

    @current_copynumbers.setter
    def current_copynumbers(self, value):
        # This may not be as fast as a[a < 0] = 0
        if value is not None:
            value[value < 0] = 0
            self._current_copynumbers = value

    @property
    def copynumbers(self):
        """ All copynumbers through time, using the convention [sample, t]"""
        return self._copynumbers

    @copynumbers.setter
    def copynumbers(self, value):
        """ Careful: does not kick in for setting slices """
        # This may not be as fast as a[a < 0] = 0
        if value is not None:
            value[value < 0] = 0
        self._copynumbers = value

    @property
    def size(self):
        return self.data.size


class BaseCircuit():
    def __init__(self, config_args):

        self.name = config_args.get("name")
        self.molecular_params = config_args['molecular_params']

        if config_args is None:
            config_args = {}

        self.species = BaseSpecies(config_args)
        self._signal = None

        self.result_collector = ResultCollector()

        self.init_graph()

    def init_graph(self):
        self._node_labels = None
        self.graph = self.build_graph()

    def build_graph(self, source_matrix=None) -> nx.DiGraph:
        if source_matrix is not None:
            inters = source_matrix
        else:
            inters = self.species.interactions
        graph = nx.from_numpy_matrix(inters, create_using=nx.DiGraph)
        if self.node_labels is not None:
            graph = nx.relabel_nodes(graph, self.node_labels)
        return graph

    def refresh_graph(self):
        self.graph = self.build_graph()

    def get_graph_labels(self) -> dict:
        return sorted(self.graph)

    def reset_to_initial_state(self):
        self.species.reset_to_initial_state()
        self.result_collector.reset()

    def make_subsystem(self, mutation_name: str, mutation=None):
        subsystem = deepcopy(self)
        subsystem.reset_to_initial_state()
        subsystem.species.loaded_interactions = False

        if mutation is None:
            mutation = self.species.mutations.get(mutation_name)

        subsystem.species.mutate(mutation)

        return subsystem

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, new_graph):
        assert type(new_graph) == nx.DiGraph, 'Cannot set graph to' + \
            f' type {type(new_graph)}.'
        self._graph = new_graph

    @property
    def node_labels(self):
        return self._node_labels

    @node_labels.setter
    def node_labels(self, labels: dict):

        current_nodes = self.get_graph_labels()
        if type(labels) == list:
            self._node_labels = dict(zip(current_nodes, labels))
        else:
            labels = list(range(len(self.species.interactions)))
            self._node_labels = dict(zip(current_nodes, labels))
        self.graph = nx.relabel_nodes(self.graph, self.node_labels)

    @property
    def results(self):
        return self.result_collector.results

    @property
    def signal(self):
        return self._signal

    @signal.getter
    def signal(self):
        if self._signal is None:
            logging.warning(f'Trying to retrieve None signal from circuit. Make sure signal specified in circuit config')
        return self._signal
    
    @signal.setter
    def signal(self, value):
        self._signal = value
