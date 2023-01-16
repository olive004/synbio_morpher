from functools import partial
import logging
import os
import pandas as pd

from fire import Fire
from src.srv.io.manage.script_manager import script_preamble, visualisation_script_protocol_preamble
from src.utils.misc.scripts_io import get_search_dir
from src.utils.misc.string_handling import prettify_keys_for_label
from src.utils.results.analytics.naming import get_true_names_analytics

from src.utils.results.experiments import Experiment, Protocol
from src.utils.results.visualisation import visualise_data
from src.utils.data.data_format_tools.common import load_json_as_dict


def main(config=None, data_writer=None):
    # Set configs
    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        # "scripts", "num_interacting", "configs", "base_config_test_2.json"))
        "scripts", "analyse_mutated_templates_loaded_2", "configs", "base_config.json"))
    config_file = load_json_as_dict(config)

    config_file, source_dirs = get_search_dir(
        config_searchdir_key='source_dirs', config_file=config_file)
    if type(source_dirs) != list:
        source_dirs = [source_dirs]

    binding_rates_threshold_upper = None
    binding_rates_threshold_upper_text = f', with cutoff at {binding_rates_threshold_upper}' if binding_rates_threshold_upper else ''

    protocols = visualisation_script_protocol_preamble(source_dirs)
    protocols.append(
        # Binding rates min interactions mutations
        Protocol(
            partial(
                visualise_data, data_writer=data_writer, cols_x=[
                    'binding_rates_dissociation_min_interaction'],
                plot_type='histplot',
                hue='mutation_num',
                out_name='binding_rates_dissociation_min_freqs_mutations_logs',
                threshold_value_max=binding_rates_threshold_upper,
                exclude_rows_zero_in_cols=['mutation_num'],
                misc_histplot_kwargs={
                    "hue": 'mutation_num',
                    "multiple": "dodge",
                    "element": "poly"},
                log_axis=(True, False),
                use_sns=True,
                title=f'Minimum ' + r'$k_d$' + ' strength',
                xlabel='Dissociation rate ' + r'$k_d$' + ' (' + r'$s^{-1}$' + ')' +
                f'{binding_rates_threshold_upper_text}'),
            req_input=True,
            name='visualise_mutated_interactions'
        ),
    )

    # Visualisations

    # Binding rates
    protocols.append(Protocol(
        partial(
            visualise_data,
            data_writer=data_writer,
            cols_x=['binding_rates_dissociation_min_interaction'],
            plot_type='histplot',
            out_name='binding_rates_dissociation_min_freqs_mutations_logs',
            threshold_value_max=binding_rates_threshold_upper,
            exclude_rows_zero_in_cols=['mutation_num'],
            misc_histplot_kwargs={
                "hue": 'mutation_num',
                "multiple": "dodge",
                "element": "poly"},
            log_axis=(True, False),
            use_sns=True,
            hue='mutation_num',
            title=f'Minimum ' + r'$k_d$' + ' strength',
            xlabel='Dissociation rate ' + r'$k_d$' + ' (' +
            r'$s^{-1}$' + ')' +
            f'{binding_rates_threshold_upper_text}'),
        req_input=True,
        name='visualise_mutated_interactions'
    ))
    # Plot mean of interacting numbers
    protocols.append(Protocol(
        partial(
            visualise_data, data_writer=data_writer,
            cols_x=['mutation_num'],
            cols_y=['num_interacting'],
            plot_type='violin_plot',
            out_name='num_interacting_barplot',
            threshold_value_max=binding_rates_threshold_upper,
            exclude_rows_zero_in_cols=['mutation_num'],
            use_sns=True,
            ci="sd",
            title=f'{prettify_keys_for_label("mutation_num")} vs. {prettify_keys_for_label("num_interacting")}',
            ylabel='Number of interactions',
            xlabel='Number of mutations'),
        req_input=True,
        name='visualise_mutated_interactions'
    ))

    def vis_analytics(data: pd.DataFrame):
        # Analytics visualisation
        analytics_types = get_true_names_analytics(data)

        # Log histplots with mutation number hue
        diff_iters = [
            [
                analytics_type,
                # f'{analytics_type}_diff_to_base_circuit',
                f'{prettify_keys_for_label(analytics_type)} difference between circuit\nand mutated counterparts',
                f'{prettify_keys_for_label(analytics_type)} difference'
            ] for analytics_type in analytics_types if 'diff' in analytics_type]
        ratio_iters = [
            [
                analytics_type,
                # f'{analytics_type}_ratio_from_mutation_to_base',
                f'{prettify_keys_for_label(analytics_type)} ratio from mutated\nto original circuit',
                f'{prettify_keys_for_label(analytics_type)} ratio'
            ] for analytics_type in analytics_types if 'ratio' in analytics_type]
        for filltype in ['dodge', 'fill']:
            for iters in diff_iters, ratio_iters:
                for cols_x, title, xlabel in iters:

                    visualise_data(
                        data=data,
                        data_writer=data_writer, cols_x=[cols_x],
                        plot_type='histplot',
                        hue='mutation_num',
                        out_name=f'{cols_x}_log_{filltype}',
                        exclude_rows_zero_in_cols=['mutation_num'],
                        misc_histplot_kwargs={
                            "multiple": filltype,
                            "hue": 'mutation_num',
                            "element": "step"
                        },
                        log_axis=(True, False),
                        use_sns=True,
                        title=title,
                        xlabel=xlabel
                    )

    protocols.append(
        Protocol(
            vis_analytics,
            req_input=True,
            name='visualise_analytics',
            skip=config_file.get('only_visualise_circuits', False)
        )
    )

    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
