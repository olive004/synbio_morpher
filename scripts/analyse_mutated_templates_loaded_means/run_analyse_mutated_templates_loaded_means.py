from functools import partial
import os
import pandas as pd

from src.srv.io.manage.script_manager import script_preamble
from src.utils.misc.io import get_pathnames_from_mult_dirs
from src.utils.misc.scripts_io import get_search_dir, load_experiment_config_original
from src.srv.sequence_exploration.summary_mutations import summarise_mutation_groups

from src.utils.results.experiments import Experiment, Protocol
from src.utils.results.result_writer import ResultWriter
from src.utils.results.visualisation import visualise_data
from src.utils.data.data_format_tools.common import load_json_as_dict, load_json_mult


def main(config=None, data_writer=None):
    # Set configs
    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
            "scripts", "analyse_mutated_templates_loaded_5", "configs", "base_config.json"))
    config_file = load_json_as_dict(config)

    # Start_experiment
    if data_writer is None:
        data_writer = ResultWriter(purpose=config_file.get(
            'experiment', {}).get('purpose', 'analyse_mutated_templates_loaded'))

    config_file, source_dirs = get_search_dir(
        config_searchdir_key='source_dirs', config_file=config_file)
    if type(source_dirs) != list:
        source_dirs = [source_dirs]

    protocols = [
        Protocol(
            partial(
                get_pathnames_from_mult_dirs,
                search_dirs=source_dirs,
                file_key='tabulated_mutation_info.json',
                first_only=True),
            req_output=True,
            name='get_pathnames_from_mult_dirs'
        ),
        Protocol(
            partial(
                load_json_mult,
                as_type=pd.DataFrame
            ),
            req_input=True,
            req_output=True,
            name='load_json'
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
        Protocol(
            summarise_mutation_groups,
            req_input=True,
            req_output=True,
            name='summarise_means'
        )
    ]


    source_config = load_experiment_config_original(
        source_dirs[0], target_purpose='mutation_effect_on_interactions_signal')
    num_mutations = source_config['mutations']['mutation_nums_within_sequence']
    num_mutations = [num_mutations] if type(
        num_mutations) != list else num_mutations
    num_mutations = num_mutations + ['all']
    log_opts = [(False, False), (True, False)]


    for s in ['mean', 'std']:
        for m in num_mutations:
            for log_opt in log_opts:
                for c in get_
    protocols.append(
        partial(visualise_data(
            data_writer=data_writer,

        )
    )

    

    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()
