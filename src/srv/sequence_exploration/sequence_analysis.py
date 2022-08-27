

import itertools
import logging
import os
import sys
import numpy as np
import pandas as pd


from src.srv.io.loaders.data_loader import GeneCircuitLoader
from src.utils.misc.string_handling import prettify_logging_info, remove_element_from_list_by_substring
from src.utils.results.writer import DataWriter
from src.srv.parameter_prediction.interactions import InteractionMatrix
from src.utils.misc.io import get_pathnames, get_subdirectories
from src.utils.misc.scripts_io import get_path_from_output_summary, get_root_experiment_folder, \
    load_experiment_config, load_experiment_output_summary, load_result_report


def generate_interaction_stats(path_name, writer: DataWriter = None, experiment_dir: str = None, **stat_addons) -> pd.DataFrame:

    interactions = InteractionMatrix(
        matrix_path=path_name, experiment_dir=experiment_dir)

    stats = interactions.get_stats()
    add_stats = pd.DataFrame.from_dict({'interactions_path': [path_name]})
    stats = pd.concat([stats, add_stats], axis=1)

    if writer:
        writer.output(out_type='csv', out_name='circuit_stats',
                      data=stats, write_master=False)

    return stats


def filter_data(data: pd.DataFrame, filters: dict = {}):
    if not filters:
        return data
    filt_stats = data[data['num_interacting']
                      >= filters.get("min_num_interacting")]
    filt_stats = filt_stats[filt_stats['num_self_interacting'] < filters.get(
        "max_self_interacting")]
    return filt_stats


def pull_circuits_from_stats(stats_pathname, filters: dict, write_key='data_path') -> list:

    stats = GeneCircuitLoader().load_data(stats_pathname).data
    filt_stats = filter_data(stats, filters)

    if filt_stats.empty:
        logging.warning(
            f'No circuits were found matching the selected filters {filters}')
        logging.warning(stats)
        return []

    experiment_folder = get_root_experiment_folder(
        filt_stats['interactions_path'].to_list()[0])
    experiment_summary = load_experiment_output_summary(experiment_folder)

    extra_configs = []
    for index, row in filt_stats.iterrows():
        extra_config = {write_key: get_path_from_output_summary(
            name=row["name"], output_summary=experiment_summary)}
        extra_config.update(
            {'interactions_path': row["interactions_path"]}
        )
        extra_config.update(load_experiment_config(experiment_folder))
        extra_configs.append(extra_config)
    # logging.info(extra_configs)
    return extra_configs


def tabulate_mutation_info(source_dir, data_writer: DataWriter):

    interaction_types = ['binding_rates', 'eqconstants', 'interactions']

    info_column_names = [
        'circuit_name',
        'mutation_name',
        'source_species',
        'mutation_num',
        'mutation_type',
        'path_to_steady_state_data',
        'path_to_signal_data',
        'path_to_template_circuit'
    ]
    info_column_names_interactions = [[f'{i}_self_interacting',
                                       f'{i}_self_interacting_diff_to_base_circuit',
                                       f'{i}_num_interacting',
                                       f'{i}_num_interacting_diff_to_base_circuit',
                                       f'{i}_max',
                                       f'{i}_max_diff_to_base_circuit'] for i in interaction_types]
    info_column_names_interactions = list(itertools.chain(*info_column_names_interactions))
    info_column_names = info_column_names + info_column_names_interactions

    logging.info(prettify_logging_info(info_column_names))
    info_table = pd.DataFrame(columns=info_column_names)

    def check_coherency(table):
        for (target, pathname) in [('circuit_name', 'path_to_template_circuit'),
                                   ('mutation_name', 'path_to_steady_state_data'), ('mutation_name', 'path_to_signal_data')]:
            if type(table) == pd.DataFrame:
                assert table[target].values[0] in table[pathname].values[0], \
                    f'Name {table[target].values[0]} should be in path {table[pathname].values[0]}.'
            else:
                assert table[target] in table[pathname], \
                    f'Name {table[target]} should be in path {table[pathname]}.'

    def make_interaction_stats(source_interaction_dir, include_circuit_in_filekey=False):
        interaction_stats = {}
        for interaction_type in interaction_types:
            interaction_dir = os.path.join(
                source_interaction_dir, interaction_type)
            file_key = [
                interaction_type, circuit_name] if include_circuit_in_filekey else interaction_type
            interaction_stats[interaction_type] = InteractionMatrix(matrix_path=get_pathnames(first_only=True,
                                                                                              file_key=file_key,
                                                                                              search_dir=interaction_dir)).get_stats()
        return interaction_stats

    def upate_table_with_results(table: dict, reference_table: dict, results: dict):
        table.update(results)
        for k in results.keys():
            reference_v = reference_table[k]
            logging.info(k)
            logging.info(np.asarray(table[k]))
            logging.info(np.asarray(reference_v))
            logging.info(reference_v)
            if type(reference_v) == bool:
                continue
            table[f'{k}_diff_to_base_circuit'] = np.asarray(table[k]) - np.asarray(reference_v)
        return table

    source_config = load_experiment_config(source_dir)

    circuit_dirs = get_subdirectories(source_dir)
    for circuit_dir in circuit_dirs:
        circuit_name = os.path.basename(circuit_dir)
        mutations_pathname = get_pathnames(
            first_only=True, file_key='mutations', search_dir=circuit_dir)
        mutations = GeneCircuitLoader().load_data(mutations_pathname).data
        mutation_dirs = sorted(get_subdirectories(circuit_dir))
        # TODO: need a better way of getting the mutation directories - maybe just create mutations in a subfolder
        for exclude_dir in ['binding_rates', 'eqconstants', '/interactions']:
            mutation_dirs = remove_element_from_list_by_substring(
                mutation_dirs, exclude=exclude_dir)

        # Unmutated circuit
        # interaction_dir = os.path.dirname(
        #     os.path.dirname(mutations['template_file'].values[0]))
        interaction_dir = circuit_dir
        interaction_stats = make_interaction_stats(
            interaction_dir, include_circuit_in_filekey=True)
            
        result_report = load_result_report(interaction_dir)

        current_og_table = {
            'circuit_name': circuit_name,
            'mutation_name': '',
            'source_species': '',
            'mutation_num': 0,
            'mutation_type': '',
            'path_to_steady_state_data': get_pathnames(first_only=True,
                                                       file_key='steady_states_data',
                                                       search_dir=circuit_dir),
            'path_to_signal_data': get_pathnames(first_only=True,
                                                 file_key='signal_data',
                                                 search_dir=circuit_dir),
            'path_to_template_circuit': ''
        }
        # Expand the interaction keys in the table
        for interaction_type in interaction_types:
            current_og_table[f'{interaction_type}_self_interacting'] = interaction_stats[interaction_type]['self_interacting']
            current_og_table[f'{interaction_type}_self_interacting_diff_to_base_circuit'] = 0
            current_og_table[f'{interaction_type}_num_interacting'] = interaction_stats[interaction_type]['num_interacting']
            current_og_table[f'{interaction_type}_num_interacting_diff_to_base_circuit'] = 0
            current_og_table[f'{interaction_type}_max'] = interaction_stats[interaction_type]['max_interaction']
            current_og_table[f'{interaction_type}_max_diff_to_base_circuit'] = 0
        # current_og_table.update(result_report)
        current_og_table = upate_table_with_results(
            current_og_table, reference_table=current_og_table, results=result_report)
        info_table = pd.concat([info_table, pd.DataFrame([current_og_table])])

        # Mutated circuits
        for mutation_dir in mutation_dirs:
            curr_mutation = mutations[mutations['mutation_name'] == os.path.basename(
                mutation_dir)]
            mutation_name = curr_mutation['mutation_name'].values[0]

            interaction_stats_current = make_interaction_stats(
                mutation_dir, include_circuit_in_filekey=False)

            current_table = {
                'circuit_name': circuit_name,
                'mutation_name': mutation_name,
                'source_species': curr_mutation['template_name'].values[0],
                'mutation_num': source_config['mutations']['mutation_nums_within_sequence'],
                'mutation_type': curr_mutation['mutation_types'].values[0],
                'path_to_steady_state_data': get_pathnames(first_only=True,
                                                           file_key='steady_states_data',
                                                           search_dir=mutation_dir),
                'path_to_signal_data': get_pathnames(first_only=True,
                                                     file_key='signal_data',
                                                     search_dir=mutation_dir),
                'path_to_template_circuit': curr_mutation['template_file'].values[0]
            }
            # Expand the interaction keys in the table
            for interaction_type in interaction_types:
                current_table[f'{interaction_type}_self_interacting'] = interaction_stats_current[interaction_type]['self_interacting']
                current_table[f'{interaction_type}_self_interacting_diff_to_base_circuit'] = interaction_stats_current[interaction_type]['self_interacting'] - \
                    interaction_stats[interaction_type]['self_interacting']
                current_table[f'{interaction_type}_count'] = interaction_stats_current[interaction_type]['num_interacting']
                current_table[f'{interaction_type}_count_diff_to_base_circuit'] = interaction_stats_current[interaction_type]['num_interacting'] - \
                    interaction_stats[interaction_type]['num_interacting']
                current_table[f'{interaction_type}_strength'] = interaction_stats_current[interaction_type]['max_interaction']
                current_table[f'{interaction_type}_strength_diff_to_base_circuit'] = interaction_stats_current[interaction_type]['max_interaction'] - \
                    interaction_stats[interaction_type]['max_interaction']
            result_report = load_result_report(mutation_dir)
            current_table = upate_table_with_results(
                current_table, reference_table=current_og_table, results=result_report)
            check_coherency(current_table)
            info_table = pd.concat([info_table, pd.DataFrame([current_table])])
    data_writer.output(
        out_type='csv', out_name='tabulated_mutation_info', **{'data': info_table})
    return info_table
