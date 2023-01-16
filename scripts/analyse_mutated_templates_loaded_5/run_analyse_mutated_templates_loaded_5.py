import operator
import os
from copy import deepcopy
import pandas as pd
from fire import Fire


from src.srv.io.manage.script_manager import script_preamble, visualisation_script_protocol_preamble
from src.utils.misc.database_handling import thresh_func
from src.utils.misc.scripts_io import get_search_dir
from src.utils.misc.string_handling import prettify_keys_for_label
from src.srv.parameter_prediction.interactions import INTERACTION_TYPES
from src.utils.results.experiments import Experiment, Protocol
from src.utils.results.analytics.naming import get_true_interaction_cols
from src.utils.results.visualisation import visualise_data
from src.utils.data.data_format_tools.common import load_json_as_dict


def main(config=None, data_writer=None):
    # Set configs
    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        "scripts", "analyse_mutated_templates_loaded_5", "configs", "base_config.json"))
    config_file = load_json_as_dict(config)

    # Visualisations

    def vis_interactions(data: pd.DataFrame):
        outlier_std_threshold_y = 3
        for interaction_type in INTERACTION_TYPES:
            cols = get_true_interaction_cols(
                data, interaction_type, remove_symmetrical=True)
            df = data.melt(id_vars='mutation_num',
                           value_vars=cols, value_name=interaction_type)
            range_df = round(df[interaction_type].max() -
                             df[interaction_type].min(), 4)
            mode = round(df[interaction_type].mode().iloc[0], 4)
            for m in list(df['mutation_num'].unique()) + ['all']:
                for log_opt in [(False, False), (True, False)]:
                    log_text = '_log' if any(log_opt) else ''
                    for thresh in ['outlier', 'exclude', 'lt', 'gt', 'lt_strict', 'gt_strict', False]:
                        if m == 'all':
                            plot_grammar_m = 's'
                            hue = 'mutation_num'
                            selection_conditions = None
                        else:
                            plot_grammar_m = 's' if m > 1 else ''
                            hue = None
                            selection_conditions = [
                                ('mutation_num', operator.eq, m)]
                        thresh_text, remove_outliers_y, selection_conditions = thresh_func(
                            thresh, range_df, mode, outlier_std_threshold_y, selection_conditions, sel_col=interaction_type)
                        for normalise in [True, False]:
                            visualise_data(
                                data=df,
                                data_writer=data_writer,
                                cols_x=[interaction_type],
                                plot_type='histplot',
                                out_name=f'{interaction_type}_norm-{normalise}{log_text}_thresh-{thresh}_m{m}',
                                log_axis=log_opt,
                                use_sns=True,
                                selection_conditions=selection_conditions,
                                remove_outliers_y=remove_outliers_y,
                                outlier_std_threshold_y=outlier_std_threshold_y,
                                hue=hue,
                                title=f'{prettify_keys_for_label(interaction_type)} for {m} mutation{plot_grammar_m}{thresh_text}',
                                xlabel=prettify_keys_for_label(
                                    interaction_type),
                                misc_histplot_kwargs={'stat': 'probability' if normalise else 'count'
                                                      # , 'hue_norm': [0, 1] if normalise else None}
                                                      }
                            )

    # Protocols
    config_file, source_dirs = get_search_dir(
        config_searchdir_key='source_dirs', config_file=config_file)
    if type(source_dirs) != list:
        source_dirs = [source_dirs]
    protocols = visualisation_script_protocol_preamble(source_dirs)
    protocols.append(
        Protocol(
            vis_interactions,
            req_input=True,
            name='vis_interactions'
        )
    )

    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
