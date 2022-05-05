

import logging
import os
import pandas as pd


from src.srv.io.loaders.data_loader import DataLoader
from src.srv.io.results.writer import DataWriter
from src.srv.parameter_prediction.interactions import InteractionMatrix
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.misc.io import get_path_from_exp_summary, get_pathnames, get_subdirectories, load_experiment_report, load_experiment_output_summary


def generate_interaction_stats(path_name, writer: DataWriter, **stat_addons):

    interactions = InteractionMatrix(matrix_path=path_name)

    stats = interactions.get_stats()
    add_stats = pd.DataFrame.from_dict({'path': [path_name]})
    stats = pd.concat([stats, add_stats], axis=1)

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


def tabulate_mutation_info(source_dir):
    # Load table of
    # - path to steady state data
    # - path to signal data
    # - path to template sequence
    # - circuit stats
    # - mutation number
    # - mutation type
    # - interaction strength
    # - interaction count
    info_table = pd.DataFrame(columns=[
        'circuit_name',
        'mutation_name',
        'source_species'
        'interaction_count',
        'interaction_strength'
        'mutation_num',
        'mutation_type',
        'path_to_steady_state_data',
        'path_to_steady_signal_data',
        'path_to_template_circuit'
    ])

    experiment_summary = load_experiment_report(source_dir)
    source_config = load_json_as_dict(experiment_summary['config_filepath'])
    circuit_stats_pathname = get_pathnames(first_only=True, file_key="circuit_stats",
                                           search_dir=source_config['source_species_templates_experiment_dir'])
    circuit_stats = DataLoader().load_data(circuit_stats_pathname).data

    circuit_dirs = get_subdirectories(source_dir)
    for circuit_dir in circuit_dirs:
        mutations_pathname = get_pathnames(
            first_only=True, file_key='mutations', search_dir=circuit_dir)
        mutations = DataLoader().load_data(mutations_pathname).data
        mutation_dirs = get_subdirectories(circuit_dir)

        info_table.append({
            'circuit_name': circuit_dir * len(mutation_dirs),
            'mutation_name': mutations['mutation_name'],
            'source_species': mutations['template_name'],
            'interaction_count': circuit_stats[circuit_stats['name']
                                               == circuit_dir]['num_interacting'] * len(mutation_dirs),
            'interaction_strength': circuit_stats[circuit_stats['name']
                                                  == circuit_dir]['max_interaction'] * len(mutation_dirs),
            'mutation_num': source_config['mutations']['mutation_nums'] * len(mutation_dirs),
            'mutation_type': mutations['mutation_types'],
            'path_to_steady_state_data': [get_pathnames(first_only=True,
                                                        file_key='steady_state_data',
                                                        search_dir=m) for m in mutation_dirs],
            'path_to_signal_data': [get_pathnames(first_only=True,
                                                  file_key='signal_data',
                                                  search_dir=m) for m in mutation_dirs],
            'path_to_template_circuit': mutations['template_file']
        })
        # for mutation_dir in mutation_dirs:
        #     current_mutation_info = mutations[mutations['mutation_name']
        #                                       == mutation_dir]
        #     info_table['circuit_name'] = circuit_dir
        #     info_table['mutation_name'] = current_mutation_info['mutation_name']
        #     info_table['source_species'] = current_mutation_info['template_name']
        #     info_table['interaction_count'] = circuit_stats[circuit_stats['name']
        #                                                     == circuit_dir]['num_interacting']
        #     info_table['interaction_strength'] = circuit_stats[circuit_stats['name']
        #                                                        == circuit_dir]['max_interaction']
        #     info_table['mutation_num'] = source_config['mutations']['mutation_nums']
        #     info_table['mutation_type'] = current_mutation_info['mutation_type']
        #     info_table['path_to_steady_state_data'] = get_pathnames(first_only=True, file_key='steady_state_data',
        #                                                             search_dir=mutation_dir)
        #     info_table['path_to_steady_state_data'] = get_pathnames(first_only=True, file_key='steady_state_data',
        #                                                             search_dir=mutation_dir)
        #     info_table['path_to_template_circuit'] = current_mutation_info['template_file']

    logging.info(info_table)
    return info_table

    # for (root, dirs, files) in os.walk(source_dir):
