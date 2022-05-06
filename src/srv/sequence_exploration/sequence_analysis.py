

import logging
import os
import pandas as pd


from src.srv.io.loaders.data_loader import DataLoader
from src.srv.io.results.writer import DataWriter
from src.srv.parameter_prediction.interactions import InteractionMatrix
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.misc.io import get_path_from_exp_summary, get_pathnames, \
    get_subdirectories, load_experiment_report, load_experiment_output_summary


def generate_interaction_stats(path_name, writer: DataWriter = None, **stat_addons):

    interactions = InteractionMatrix(matrix_path=path_name)

    stats = interactions.get_stats()
    add_stats = pd.DataFrame.from_dict({'path': [path_name]})
    stats = pd.concat([stats, add_stats], axis=1)

    if writer:
        writer.output(out_type='csv', out_name='circuit_stats',
                      data=stats, write_master=False)

    return stats


def pull_circuits_from_stats(stats_pathname, filters: dict, write_key='data_path') -> list:

    stats = DataLoader().load_data(stats_pathname).data

    filt_stats = stats[stats['num_interacting']
                       >= filters.get("min_num_interacting")]
    filt_stats = filt_stats[filt_stats['num_self_interacting'] < filters.get(
        "max_self_interacting")]

    base_folder = os.path.dirname(
        os.path.dirname(filt_stats['path'].to_list()[0]))
    experiment_summary = load_experiment_output_summary(base_folder)
    experiment_report = load_experiment_report(base_folder)

    extra_configs = []
    for index, row in filt_stats.iterrows():
        extra_config = {"data_path": get_path_from_exp_summary(
            row["name"], experiment_summary)}
        extra_config.update(
            {'interactions_path': row["path"]}
        )
        extra_config.update(load_json_as_dict(
            experiment_report['config_filepath']))
        extra_configs.append(extra_config)
    return extra_configs


def tabulate_mutation_info(source_dir, data_writer: DataWriter):

    info_table = pd.DataFrame(columns=[
        'circuit_name',
        'mutation_name',
        'source_species',
        'interaction_count',
        'interaction_strength',
        'interaction_strength_diff_to_base_circuit',
        'mutation_num',
        'mutation_type',
        'path_to_steady_state_data',
        'path_to_signal_data',
        'path_to_template_circuit'
    ])

    def check_coherency(table):
        for (target, pathname) in [('circuit_name', 'path_to_template_circuit'),
                                   ('mutation_name', 'path_to_steady_state_data'), ('mutation_name', 'path_to_signal_data')]:
            if type(table) == pd.DataFrame:
                assert table[target].values[0] in table[pathname].values[0], \
                    f'Name {table[target].values[0]} should be in path {table[pathname].values[0]}.'
            else:
                assert table[target] in table[pathname], \
                    f'Name {table[target]} should be in path {table[pathname]}.'

    experiment_summary = load_experiment_report(source_dir)
    source_config = load_json_as_dict(experiment_summary['config_filepath'])
    # circuit_stats_pathname = get_pathnames(first_only=True, file_key="circuit_stats",
    #                                        search_dir=source_config['source_species_templates_experiment_dir'])
    # circuit_stats = DataLoader().load_data(circuit_stats_pathname).data

    circuit_dirs = get_subdirectories(source_dir)
    for circuit_dir in circuit_dirs:
        mutations_pathname = get_pathnames(
            first_only=True, file_key='mutations', search_dir=circuit_dir)
        mutations = DataLoader().load_data(mutations_pathname).data
        mutation_dirs = sorted(get_subdirectories(circuit_dir))

        circuit_name = os.path.basename(circuit_dir)

        # Unmutated circuit
        interaction_dir = os.path.join(
            os.path.dirname(os.path.dirname(mutations['template_file'].values[0])), 'interactions')
        interaction_stats = InteractionMatrix(matrix_path=get_pathnames(first_only=True,
                                                                        file_key="interactions",
                                                                        search_dir=interaction_dir)).get_stats()
        circuit_interaction_max = interaction_stats['max_interaction']
        current_table = pd.DataFrame.from_dict({
            'circuit_name': circuit_name,
            'mutation_name': '',
            'source_species': '',
            'interaction_count': interaction_stats['num_interacting'],
            'interaction_strength': circuit_interaction_max,
            'interaction_strength_diff_to_base_circuit': 0,
            'mutation_num': 0,
            'mutation_type': '',
            'path_to_steady_state_data': get_pathnames(first_only=True,
                                                       file_key='steady_state_data',
                                                       search_dir=circuit_dir),
            'path_to_signal_data': get_pathnames(first_only=True,
                                                 file_key='signal_data',
                                                 search_dir=circuit_dir),
            'path_to_template_circuit': ''
        })
        info_table = pd.concat([info_table, current_table])

        # Mutated circuits
        for mutation_dir in mutation_dirs:
            curr_mutation = mutations[mutations['mutation_name'] == os.path.basename(
                mutation_dir)]
            mutation_name = curr_mutation['mutation_name'].values[0]
            interaction_stats = InteractionMatrix(matrix_path=get_pathnames(first_only=True,
                                                                            file_key="interactions",
                                                                            search_dir=os.path.join(
                                                                                mutation_dir, 'interactions'))).get_stats()
            current_table = pd.DataFrame.from_dict({
                'circuit_name': circuit_name,
                'mutation_name': mutation_name,
                'source_species': curr_mutation['template_name'].values[0],
                'interaction_count': interaction_stats['num_interacting'],
                'interaction_strength': interaction_stats['max_interaction'],
                'interaction_strength_diff_to_base_circuit': circuit_interaction_max - interaction_stats['max_interaction'],
                'mutation_num': source_config['mutations']['mutation_nums'],
                'mutation_type': curr_mutation['mutation_types'].values[0],
                'path_to_steady_state_data': get_pathnames(first_only=True,
                                                           file_key='steady_state_data',
                                                           search_dir=mutation_dir),
                'path_to_signal_data': get_pathnames(first_only=True,
                                                     file_key='signal_data',
                                                     search_dir=mutation_dir),
                'path_to_template_circuit': curr_mutation['template_file'].values[0]
            })
            check_coherency(current_table)
            info_table = pd.concat([info_table, current_table])
    data_writer.output(
        out_type='csv', out_name='tabulated_mutation_info', **{'data': info_table})
    return info_table
