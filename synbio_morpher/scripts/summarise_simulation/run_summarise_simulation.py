
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
from functools import partial
import logging
import os
import pandas as pd

from fire import Fire
from src.srv.io.manage.script_manager import script_preamble
from src.utils.misc.scripts_io import get_search_dir

from src.utils.results.experiments import Experiment, Protocol
from src.utils.results.result_writer import ResultWriter
from src.srv.sequence_exploration.sequence_analysis import b_tabulate_mutation_info
from src.utils.data.data_format_tools.common import load_json_as_dict, load_multiple_as_list


def main(config=None, data_writer=None):
    # Set configs
    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
            # "src", "scripts", "summarise_simulation", "configs", "analyse_mutated_templates_20_highmag.json"))
            "src", "scripts", "summarise_simulation", "configs", "base_config.json"))
    config_file = load_json_as_dict(config)

    # Start_experiment
    if data_writer is None:
        data_writer = ResultWriter(purpose=config_file.get('experiment', {}).get('purpose'))

    config_file, source_dirs = get_search_dir(
        config_searchdir_key='source_dirs', config_file=config_file)
    if type(source_dirs) != list:
        source_dirs = [source_dirs]
    protocols = [
        Protocol(
            partial(load_multiple_as_list, inputs_list=source_dirs, load_func=b_tabulate_mutation_info, 
            data_writer=data_writer),
            req_output=True,
            name='b_tabulate_mutation_info'
        ),
        Protocol(
            partial(
                pd.concat,
                axis=0,
                ignore_index=True),
            req_input=True,
            req_output=True,
            name='concatenate_dfs'
        ),
    ]
    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
