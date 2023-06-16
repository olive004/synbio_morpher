
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
from copy import deepcopy
from functools import partial
import logging
import os
from src.scripts.parameter_grid_analysis.run_parameter_grid_analysis import main as parameter_grid_analysis
from src.srv.io.manage.script_manager import script_preamble
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.misc.type_handling import inverse_dict
from src.utils.parameter_inference.interpolation_grid import create_parameter_range
from src.utils.results.experiments import Experiment, Protocol


def main(config=None, data_writer=None):

    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        'src', 'scripts', 'parameter_grid_analysis', 'configs', 'testing.json'))
    config_file = load_json_as_dict(config)

    def hash_species_in_cfg(config):
        for k, v in config["slicing"]["interactions"].items():
            if 'species' in k and type(v) == dict:
                config["slicing"]["interactions"][k] = {
                    k_: tuple(v_) for k_, v_ in config["slicing"]["interactions"][k].items()}
        return config

    def loop_parameter_grid_analysis(config):
        mod_config = hash_species_in_cfg(deepcopy(config))
        mod_config["slicing"]["interactions"]["multi_varying_strengths"] = {k: create_parameter_range(v) for k, v in
                                                                            mod_config["slicing"]["interactions"]['multi_varying_strengths'].items()}
        interactions_cfg = mod_config["slicing"]["interactions"]
        inverse_non_varying_cfg = inverse_dict(
            mod_config["slicing"]["interactions"]["non_varying_species_interactions"])

        for label, non_varying_species in interactions_cfg['multi_varying_species_interactions'].items():
            data_writer.subdivide_writing(str(non_varying_species))
            if non_varying_species in inverse_non_varying_cfg:
                arbitrary_species_key = inverse_non_varying_cfg[non_varying_species]
            else:
                arbitrary_species_key = int(max(list(
                    mod_config["slicing"]["interactions"]["non_varying_species_interactions"].keys()))) + 1
            non_varying_strengths = create_parameter_range(
                mod_config["slicing"]["interactions"]['multi_varying_strengths'][label])
            for non_varying_strength in non_varying_strengths:

                logging.info(arbitrary_species_key)
                logging.info(non_varying_species)
                logging.info(non_varying_strength)
                config["slicing"]["interactions"]["non_varying_species_interactions"][arbitrary_species_key] = list(
                    non_varying_species)
                config["slicing"]["interactions"]["non_varying_strengths"][arbitrary_species_key] = non_varying_strength
                parameter_grid_analysis(config, data_writer)
                data_writer.unsubdivide()

    experiment = Experiment(config=config, config_file=config_file, protocols=[
        Protocol(partial(loop_parameter_grid_analysis, config=config_file))
    ], data_writer=data_writer)
    experiment.run_experiment()
