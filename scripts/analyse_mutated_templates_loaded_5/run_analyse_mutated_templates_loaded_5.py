import operator
import os
import pandas as pd
from fire import Fire


from src.srv.io.manage.script_manager import script_preamble, visualisation_script_protocol_preamble
from src.utils.misc.scripts_io import get_search_dir
from src.utils.misc.string_handling import prettify_keys_for_label
from src.srv.parameter_prediction.interactions import INTERACTION_TYPES
from src.utils.results.experiments import Experiment, Protocol
from src.utils.results.visualisation import visualise_data
from src.utils.data.data_format_tools.common import load_json_as_dict


def main(config=None, data_writer=None):
    # Set configs
    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        "scripts", "analyse_mutated_templates_loaded_5", "configs", "base_config.json"))
    config_file = load_json_as_dict(config)

    # Visualisations
    def get_true_interaction_cols(data, interaction_attr):
        num_species = len(data['sample_names'].unique())
        names = []
        for i in range(num_species):
            for ii in range(num_species):
                names.append(interaction_attr + '_' +
                             str(i) + '-' + str(ii))
        assert all([n in data.columns for n in names]
                   ), f'Interaction info column names were not isolated correctly: {names}'
        return names

    def vis_interactions(data: pd.DataFrame):

        for interaction_type in INTERACTION_TYPES:
            cols = get_true_interaction_cols(data, interaction_type)
            for log_opt in [(False, False), (True, False)]:
                log_text = '' if all(log_opt) else '_log'
                for m in list(data['mutation_num'].unique()) + ['all']:
                    if m == 'all':
                        plot_grammar_m = 's'
                        hue = 'mutation_num'
                        selection_conditions = None
                    else:
                        plot_grammar_m = 's' if m > 1 else ''
                        hue = None
                        selection_conditions = [
                            ('mutation_num', operator.eq, m)]

                    visualise_data(
                        og_data=data,
                        data_writer=data_writer,
                        cols_x=cols,
                        plot_type='bar_plot',
                        out_name=f'{interaction_type}{log_text}_all',
                        log_axis=log_opt,
                        use_sns=True,
                        selection_conditions=selection_conditions,
                        hue=hue,
                        title=f'{interaction_type} for {m} mutation{plot_grammar_m}',
                        xlabel=prettify_keys_for_label(interaction_type),
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
