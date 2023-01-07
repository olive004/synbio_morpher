from functools import partial
import logging
import os
import operator

import numpy as np
import pandas as pd

from fire import Fire
from src.srv.io.manage.script_manager import script_preamble
from src.utils.misc.io import get_pathnames_from_mult_dirs
from src.utils.misc.scripts_io import get_search_dir, load_experiment_config_original
from src.utils.misc.string_handling import prettify_keys_for_label
from src.utils.results.analytics.naming import get_analytics_types, get_signal_dependent_analytics, DIFF_KEY, RATIO_KEY

from src.utils.results.experiments import Experiment, Protocol
from src.utils.results.result_writer import ResultWriter
from src.utils.results.visualisation import visualise_data
from src.srv.parameter_prediction.interactions import INTERACTION_TYPES
from src.srv.parameter_prediction.simulator import SIMULATOR_UNITS
from src.utils.data.data_format_tools.common import load_json_as_dict, load_csv_mult


def main(config=None, data_writer=None):
    # Set configs
    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        # "scripts", "analyse_mutated_templates_loaded", "configs", "base_config_test_2.json"))
        # "scripts", "analyse_mutated_templates_loaded", "configs", "base_config.json"))
        # "scripts", "analyse_mutated_templates_loaded", "configs", "analyse_large.json"))
        # "scripts", "analyse_mutated_templates_loaded", "configs", "analyse_large_highmag.json"))
        "scripts", "analyse_mutated_templates_loaded_1", "configs", "base_config.json"))
    config_file = load_json_as_dict(config)

    # Start_experiment
    if data_writer is None:
        data_writer = ResultWriter(purpose=config_file.get(
            'experiment', {}).get('purpose', 'analyse_mutated_templates_loaded_1'))

    config_file, source_dirs = get_search_dir(
        config_searchdir_key='source_dirs', config_file=config_file)
    if type(source_dirs) != list:
        source_dirs = [source_dirs]
    source_dir = source_dirs[0]
    source_config = load_experiment_config_original(
        source_dir, 'mutation_effect_on_interactions_signal')

    # binding_rates_threshold_upper = np.power(10, 6)
    binding_rates_threshold_upper = None
    binding_rates_threshold_upper_text = f', with cutoff at {binding_rates_threshold_upper}' if binding_rates_threshold_upper else ''

    def readout(v):
        logging.info(v)
        return v

    protocols = [
        Protocol(
            partial(
                get_pathnames_from_mult_dirs,
                search_dirs=source_dirs,
                file_key='tabulated_mutation_info.csv',
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
        )
    ]

    # Visualisations
    def visualise_interactions_raw(data: pd.DataFrame, interaction_types, data_writer):
        log_opts = [(False, False), (True, False)]
        num_mutations = list(data['mutation_num'].unique())
        for interaction_type in interaction_types:
            interaction_cols = [
                c for c in data.columns if interaction_type in c and DIFF_KEY not in c and RATIO_KEY not in c and 'max' not in c and 'min' not in c]
            if not interaction_cols:
                continue
            units_text = f'({SIMULATOR_UNITS[source_config["interaction_simulator"]["name"]]["rate"]}) ' if 'rate' in interaction_type else ''
            for log_opt in log_opts:
                log_text = '_log' if any(log_opt) else ''
                for m in num_mutations+['all'] + ['all-pooled']:
                    df = pd.concat(objs=[
                        pd.DataFrame.from_dict(
                            {interaction_type: data[interaction_col],
                             'mutation_num': data['mutation_num']}
                        ) for interaction_col in interaction_cols
                    ])
                    if 'all' in str(m):
                        plot_grammar_m = 's'
                        hue = 'mutation_num'
                        selection_conditions = None
                        if m == 'all-pooled':
                            hue = None
                    else:
                        plot_grammar_m = 's' if m > 1 else ''
                        hue = None
                        selection_conditions = [
                            ('mutation_num', operator.eq, m)]

                    visualise_data(
                        og_data=df,
                        cols_x=[interaction_type],
                        plot_type='histplot',
                        data_writer=data_writer,
                        out_name=interaction_type + log_text + '_m' + str(m),
                        hue=hue,
                        selection_conditions=selection_conditions,
                        log_axis=log_opt,
                        use_sns=True,
                        title=f'{prettify_keys_for_label(interaction_type)} {units_text}for {m} mutation{plot_grammar_m}'
                    )

    protocols.append(
        Protocol(
            partial(
                visualise_interactions_raw,
                interaction_types=INTERACTION_TYPES,
                data_writer=data_writer
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
