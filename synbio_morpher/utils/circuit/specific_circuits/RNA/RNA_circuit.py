
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
import logging
from synbio_morpher.utils.circuit.agnostic_circuits.circuit import Circuit

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
FORMAT = "%(filename)s:%(funcName)s():%(lineno)i: %(message)s %(levelname)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# class RNACircuit(BaseCircuit):
#     def __init__(self, config_args):
#         super(RNACircuit, self).__init__(config_args)

#         self.simulator_args = config_args['interaction_simulator']

#         self.species = self.init_species(config_args)
#         self.process_species()

#         # Rates
#         # 3 h^-1 or 20 mins or 1200s - Dilution rate
#         starting_copynumbers = self.molecular_params['starting_copynumbers']
#         self.transcription_rate = self.molecular_params['creation_rate']
#         self.degradation_rate = self.molecular_params['degradation_rate']

#         self.species.copynumbers, \
#             self.species.degradation_rates, \
#             self.species.creation_rates = self.species.init_matrices(ndims=1, init_type="uniform",
#                                                                      uniform_vals=[starting_copynumbers,
#                                                                                    self.degradation_rate,
#                                                                                    self.transcription_rate])

#     def init_species(self, args):
#         return RNASpecies(args)

#     def process_species(self):
#         self.node_labels = self.species.data.sample_names
