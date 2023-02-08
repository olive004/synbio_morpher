from functools import partial
import os
from fire import Fire
import operator
import pandas as pd

from src.srv.io.manage.script_manager import script_preamble, visualisation_script_protocol_preamble
from src.srv.parameter_prediction.interactions import INTERACTION_TYPES
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.misc.database_handling import thresh_func, select_rows_by_conditional_cols
from src.utils.misc.string_handling import prettify_keys_for_label
from src.utils.misc.scripts_io import get_search_dir
from src.utils.results.analytics.naming import get_true_names_analytics, get_true_interaction_cols
from src.utils.results.experiments import Experiment, Protocol
from src.utils.results.result_writer import ResultWriter
from src.utils.results.visualisation import visualise_data


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

    protocols = visualisation_script_protocol_preamble(source_dirs)

    def visualise_all(data: pd.DataFrame, data_writer, remove_noninteracting: bool = True):
        cols_analytics = get_true_names_analytics(data)
        for interaction_type in INTERACTION_TYPES:
            cols = get_true_interaction_cols(
                data, interaction_type, remove_symmetrical=True)

            # Remove duplicates
            df = data[data['sample_name'] == data['sample_name'].iloc[0]]
            df = df.melt(id_vars=['circuit_name', 'mutation_num'],
                         value_vars=cols, value_name=interaction_type)
            if remove_noninteracting:
                noninteracting_val = 1 if interaction_type == 'eqconstants' else 0.0015095809699999998
                df = df[df[interaction_type] != noninteracting_val]
            visualise_means_std(df, [interaction_type], data_writer)
        visualise_means_std(data, cols_analytics, data_writer)

    def visualise_means_std(data: pd.DataFrame, cols: list, data_writer):
        """ Data from is multi index """
        log_opts = [(False, False), (True, False)]
        outlier_std_threshold_y = 3

        for c in cols:
            range_df = round(data[c].max() -
                             data[c].min(), 4)
            mode = round(data[c].mode().iloc[0], 4)
            for m in list(data['mutation_num'].unique()) + ['all']:
                for s in ['mean', 'std']:
                    if m == 'all':
                        hue = 'mutation_num'
                        selection_conditions = None
                    else:
                        hue = None
                        selection_conditions = [(
                            'mutation_num', operator.eq, m
                        )]
                    for thresh in [False, 'outlier', 'lt', 'gt', 'lt_strict', 'gt_strict', 'exclude']:
                        thresh_text, remove_outliers_y, selection_conditions = thresh_func(
                            thresh, range_df, mode, outlier_std_threshold_y, selection_conditions, sel_col=c)

                        if selection_conditions:
                            data_selected = select_rows_by_conditional_cols(
                                data, selection_conditions)
                        else:
                            data_selected = data
                        d = data_selected.groupby(['circuit_name', 'mutation_num'], as_index=False).agg(
                            {c: ('mean', 'std') for c in cols})
                        for log_opt in log_opts:
                            log_text = '_log' if any(log_opt) else ''
                            for normalise in [True, False]:
                                visualise_data(
                                    data=d,
                                    data_writer=data_writer,
                                    cols_x=[(c, s)],
                                    plot_type='histplot',
                                    out_name=f'{c}_norm-{normalise}_thresh-{thresh}{log_text}_m{m}_{s}',
                                    hue=hue,
                                    use_sns=True,
                                    log_axis=log_opt,
                                    # selection_conditions=selection_conditions,
                                    remove_outliers_y=remove_outliers_y,
                                    outlier_std_threshold_y=outlier_std_threshold_y,
                                    xlabel=(
                                        f'{prettify_keys_for_label(c)}', f'{s}'),
                                    title=f'{prettify_keys_for_label(s)} of {prettify_keys_for_label(c)}\n for {m} mutations{thresh_text}',
                                    misc_histplot_kwargs={'stat': 'probability' if normalise else 'count'
                                                          # 'hue_norm': [0, 1] if normalise else None
                                                          }
                                )

    protocols.append(
        Protocol(
            partial(visualise_all,
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
