from functools import partial
import os
from fire import Fire
import operator
import pandas as pd

from src.srv.io.manage.script_manager import script_preamble
from src.utils.misc.io import get_pathnames_from_mult_dirs
from src.utils.misc.string_handling import prettify_keys_for_label
from src.utils.misc.scripts_io import get_search_dir, load_experiment_config_original
from src.utils.results.analytics.analytics import get_true_names_analytics
from src.srv.sequence_exploration.summary_mutations import summarise_mutation_groups

from src.utils.results.experiments import Experiment, Protocol
from src.utils.results.result_writer import ResultWriter
from src.utils.results.visualisation import visualise_data
from src.utils.data.data_format_tools.common import load_json_as_dict, load_csv_mult


def main(config=None, data_writer=None):
    # Set configs
    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        "scripts", "analyse_mutated_templates_loaded_means", "configs", "base_config.json"))
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
                load_csv_mult
                # as_type=pd.DataFrame
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

    # source_config = load_experiment_config_original(
    #     source_dirs[0], target_purpose='mutation_effect_on_interactions_signal')
    # num_mutations = source_config['mutations']['mutation_nums_within_sequence']
    # num_mutations = [num_mutations] if type(
    #     num_mutations) != list else num_mutations
    # num_mutations = num_mutations + ['all']

    def visualise_means_std(data: pd.DataFrame, data_writer):
        """ Data from is multi index """
        log_opts = [(False, False), (True, False)]
        for m in list(data['mutation_num'].unique()) + ['all']:
            if m == 'all':
                hue = 'mutation_num'
                selection_conditions = None
            else:
                hue = None
                selection_conditions = [(
                    'mutation_num', operator.eq, m
                )]
            for log_opt in log_opts:
                log_text = '_log'
                for c in get_true_names_analytics([c[0] for c in data.columns]):
                    for s in ['mean', 'std']:
                        visualise_data(
                            og_data=data,
                            data_writer=data_writer,
                            cols_x=[(c, s)],
                            plot_type='histplot',
                            out_name=f'{c}_{log_text}_m{m}_{s}',
                            hue=hue,
                            use_sns=True,
                            log_axis=log_opt,
                            selection_conditions=selection_conditions,
                            xlabel=(f'{prettify_keys_for_label(c)}', f'{s}'),
                            title=f'{prettify_keys_for_label(s)} of {prettify_keys_for_label(c)}\n for {m} mutations'
                        )

    protocols.append(
        Protocol(
            partial(visualise_means_std,
                    data_writer=data_writer,
                    ),
            req_input=True,
            name='visualise'
        )
    )

    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
