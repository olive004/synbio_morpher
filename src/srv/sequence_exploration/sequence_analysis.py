

from typing import List, Dict
import logging
import os
import numpy as np
import pandas as pd
from bioreaction.model.data_tools import construct_model_fromnames


from src.srv.io.loaders.data_loader import GeneCircuitLoader
from src.utils.data.data_format_tools.common import load_csv_mult
from src.utils.misc.numerical import NUMERICAL, cast_astype
from src.utils.misc.io import get_pathnames, get_subdirectories, get_pathnames_from_mult_dirs
from src.utils.misc.scripts_io import get_path_from_output_summary, get_root_experiment_folder, \
    load_experiment_config, load_experiment_output_summary, load_result_report
from src.utils.misc.type_handling import flatten_nested_listlike
from src.utils.results.analytics.naming import get_analytics_types_all, DIFF_KEY, RATIO_KEY
from src.utils.misc.database_handling import expand_data_by_col
from src.utils.results.writer import DataWriter
from src.srv.parameter_prediction.interactions import InteractionMatrix, b_get_stats, INTERACTION_TYPES, INTERACTION_FIELDS_TO_WRITE


INTERACTION_STATS = ['max_interaction',
                     'min_interaction']
MUTATION_INFO_COLUMN_NAMES = [
    'circuit_name',
    'mutation_name',
    'mutation_num',
    'mutation_type',
    'mutation_positions',
    'path_to_template_circuit'
]


def load_tabulated_info(source_dirs: list):

    info_pathnames = get_pathnames_from_mult_dirs(
        search_dirs=source_dirs,
        file_key='tabulated_mutation_info.csv',
        first_only=True)
    info = load_csv_mult(info_pathnames)
    info = pd.concat(info,
                     axis=0,
                     ignore_index=True)
    return info


def generate_interaction_stats(path_name: dict, writer: DataWriter = None, experiment_dir: str = None, **stat_addons) -> pd.DataFrame:

    interactions = 1
    interactions = InteractionMatrix(
        matrix_paths=path_name, experiment_dir=experiment_dir)

    stats = interactions.get_stats()
    add_stats = pd.DataFrame.from_dict(
        {f'path_{k}': [v] for k, v in path_name.items()})
    stats = pd.concat([stats, add_stats], axis=1)

    if writer:
        writer.output(out_type='csv', out_name='circuit_stats',
                      data=stats, write_master=False)

    return stats


def filter_data(data: pd.DataFrame, filters: dict = {}):
    if not filters:
        return data
    fks = list(filters.keys())
    for k in fks:
        if filters[k] is None:
            filters.pop(k)
    filt_stats = data
    if filters.get("min_num_interacting") is not None:
        filt_stats = data[data['num_interacting']
                          >= filters["min_num_interacting"]]
    if filters.get("max_self_interacting") is not None:
        filt_stats = filt_stats[filt_stats['num_self_interacting'] <= filters.get(
            "max_self_interacting", data["num_self_interacting"].iloc[0])]
    filt_stats = filt_stats.iloc[:min(filters.get(
        'max_total', len(filt_stats)), len(filt_stats))]
    return filt_stats


def pull_circuits_from_stats(stats_pathname, filters: dict, write_key='data_path') -> List[Dict]:

    stats = GeneCircuitLoader().load_data(stats_pathname).data
    filt_stats = filter_data(stats, filters)

    if filt_stats.empty:
        logging.warning(
            f'No circuits were found matching the selected filters {filters}')
        logging.warning(stats)
        return []

    interaction_path_keys = {
        k: f'path_{k}' for k in INTERACTION_TYPES}
    some_pathname = filt_stats[list(interaction_path_keys.values())[
        0]].to_list()[0]
    experiment_folder = get_root_experiment_folder(some_pathname)
    experiment_summary = load_experiment_output_summary(experiment_folder)

    extra_configs = []
    for index, row in filt_stats.iterrows():
        extra_config = load_experiment_config(experiment_folder)
        extra_config.update({write_key: get_path_from_output_summary(
            name=row["name"], output_summary=experiment_summary)})
        extra_config["interactions"] = {
            k: row[path_key] for k, path_key in interaction_path_keys.items()
        }
        extra_configs.append(extra_config)
    if filters.get('max_circuits') is not None:
        extra_configs = extra_configs[:filters.get('max_circuits')]

    return extra_configs


def get_mutation_info_columns(use_stats=False):
    info_column_names = MUTATION_INFO_COLUMN_NAMES
    if use_stats:
        # Difference
        info_column_names_interactions_diff = [[[f'{i}_{s}',
                                                f'{i}_{s}{DIFF_KEY}'] for s in INTERACTION_STATS] for i in INTERACTION_TYPES]
        info_column_names_interactions_diff = flatten_nested_listlike(
            info_column_names_interactions_diff)

        # Ratio
        info_column_names_interactions_ratio = [
            [[f'{i}_{s}{RATIO_KEY}'] for s in INTERACTION_STATS] for i in INTERACTION_TYPES]
        info_column_names_interactions_ratio = flatten_nested_listlike(
            info_column_names_interactions_ratio)
        info_column_names = MUTATION_INFO_COLUMN_NAMES + \
            info_column_names_interactions_diff + info_column_names_interactions_ratio
    return info_column_names


def b_tabulate_mutation_info(source_dir: str, data_writer: DataWriter, experiment_config: dict = None) -> pd.DataFrame:

    def init_info_table() -> pd.DataFrame:
        info_column_names = get_mutation_info_columns()
        info_table = pd.DataFrame(columns=info_column_names, dtype=object)
        return info_table

    def check_coherency(table: pd.DataFrame) -> None:
        for (target, pathname) in [('circuit_name', 'path_to_template_circuit')]:
            if type(table) == pd.DataFrame:
                assert table[target].values[0] in table[pathname].values[0], \
                    f'Name {table[target].values[0]} should be in path {table[pathname].values[0]}.'
            else:
                assert table[target] in table[pathname], \
                    f'Name {table[target]} should be in path {table[pathname]}.'

    def make_interaction_stats_and_sample_names(source_interaction_dirs: List[str], experiment_config: dict = None):
        interaction_matrices = [None] * len(source_interaction_dirs)
        for i, source_interaction_dir in enumerate(source_interaction_dirs):
            matrix_paths = get_pathnames(file_key=INTERACTION_TYPES,
                                         search_dir=source_interaction_dir,
                                         subdirs=INTERACTION_TYPES,
                                         as_dict=True, first_only=True)
            matrix_paths.update(get_pathnames(file_key=[INTERACTION_FIELDS_TO_WRITE[0]],
                                              search_dir=source_interaction_dir,
                                              subdirs=[
                                                  INTERACTION_FIELDS_TO_WRITE[0]],
                                              as_dict=True, first_only=True, allow_empty=True))
            interaction_matrices[i] = InteractionMatrix(
                matrix_paths=matrix_paths,
                experiment_config=experiment_config
            )
        return b_get_stats(interaction_matrices), interaction_matrices[0].sample_names

    def update_diff_to_base_circuit(stats: pd.DataFrame,
                                    ref_stats: pd.DataFrame, diffable_cols: list) -> pd.DataFrame:

        diff_table = ref_stats[diffable_cols].values.squeeze(
        ) - stats[diffable_cols]
        diff_table = diff_table.rename(
            columns={i: i + DIFF_KEY for i in diffable_cols})

        ratio_table = stats[diffable_cols] / \
            np.where(ref_stats[diffable_cols].values.squeeze() != 0,
                     ref_stats[diffable_cols].values.squeeze(), np.inf)
        ratio_table = ratio_table.rename(
            columns={i: i + RATIO_KEY for i in diffable_cols})

        return diff_table, ratio_table

    def update_info_table(info_table: pd.DataFrame, curr_table: pd.DataFrame, int_stats: pd.DataFrame,
                          ref_stats: pd.DataFrame, result_source_dirs: List[str],
                          # Needed to load the right results
                          all_sample_names: List[str], chosen_sample_names: List[str],
                          check_coherent: bool = False) -> pd.DataFrame:
        diff_cols = ['num_self_interacting',
                     'num_interacting',
                     'max_interaction',
                     'min_interaction']
        diff_interactions, ratio_interactions = update_diff_to_base_circuit(
            int_stats, ref_stats, diffable_cols=[c for c in int_stats.columns if any(
                [col in c for col in diff_cols])])

        def make_result_report(source_dir, all_sample_names, chosen_sample_names):
            result = pd.DataFrame.from_dict(
                load_result_report(source_dir, result_type='signal', index=list(
                    map(all_sample_names.index, chosen_sample_names)))
            )
            result['sample_name'] = chosen_sample_names
            result['result_source_dir'] = source_dir
            return result

        result_reports = []
        for result_source_dir in result_source_dirs:
            result_reports.append(
                make_result_report(result_source_dir,
                                   all_sample_names, chosen_sample_names)
            )
        result_table = pd.concat(result_reports, axis=0)

        def process_results(table):
            """ Contracts the columns of the table into list (reverse of explode) """
            table = table.groupby('result_source_dir').agg(
                {c: lambda x: x.tolist() for c in table.columns})
            table = table.drop(columns=['result_source_dir']).reset_index().drop(
                columns=['result_source_dir'])
            return table

        # Contract columns so that the results match the length of the current table
        result_table = process_results(result_table)

        new_table = pd.concat([pd.DataFrame.from_dict(
            curr_table), int_stats, diff_interactions, ratio_interactions, result_table], axis=1)
        new_table = new_table.explode(list(result_table.columns))
        if check_coherent:
            check_coherency(new_table)
        info_table = pd.concat([info_table, new_table])
        return info_table

    def prepare_for_writing(info_table: pd.DataFrame) -> pd.DataFrame:
        result_report_keys = get_analytics_types_all()
        info_table = expand_data_by_col(info_table, columns=result_report_keys, find_all_similar_columns=True,
                                        column_for_expanding_coldata='sample_names', idx_for_expanding_coldata=0)
        return info_table

    def write_table(info_table: pd.DataFrame) -> None:
        data_writer.output(
            out_type='csv', out_name='tabulated_mutation_info', **{'data': info_table})

    def write_results(info_table: pd.DataFrame) -> None:
        info_table = prepare_for_writing(info_table)
        write_table(info_table)

    info_table = init_info_table()
    circuit_dirs = get_subdirectories(source_dir, min_condition=3)
    for circ_idx, circuit_dir in enumerate(circuit_dirs):
        circuit_name = os.path.basename(circuit_dir)

        interaction_stats_og, sample_names = make_interaction_stats_and_sample_names(
            [circuit_dir], experiment_config=experiment_config)
        all_sample_names = [
            s.name for s in construct_model_fromnames(sample_names).species]

        current_og_table = {
            'circuit_name': circuit_name,
            'mutation_name': 'ref_circuit',
            'mutation_num': 0,
            'mutation_type': [],
            'mutation_positions': [],
            'path_to_template_circuit': ''
        }
        current_og_table = pd.DataFrame.from_dict(
            {k: [v] for k, v in current_og_table.items()})
        # Expand the interaction keys in the table
        info_table = update_info_table(info_table, curr_table=current_og_table, int_stats=interaction_stats_og,
                                       ref_stats=interaction_stats_og, result_source_dirs=[
                                           circuit_dir],
                                       all_sample_names=all_sample_names, chosen_sample_names=sample_names)

        mutations_pathname = get_pathnames(
            first_only=True, file_key='mutations', search_dir=circuit_dir, allow_empty=True)
        if mutations_pathname:
            mutations = GeneCircuitLoader().load_data(mutations_pathname).data
            if os.path.isdir(os.path.join(circuit_dir, 'mutations')):
                mutation_dirs = sorted(get_subdirectories(
                    os.path.join(circuit_dir, 'mutations')))
            else:
                mutation_dirs = sorted(get_subdirectories(circuit_dir))

            interaction_stats = make_interaction_stats_and_sample_names(
                mutation_dirs, experiment_config=experiment_config)[0]

            # Mutated circuits
            mutation_table = pd.DataFrame.from_dict({
                'circuit_name': [circuit_name] * len(mutation_dirs),
                'mutation_name': mutations['mutation_name'],
                'mutation_num': mutations['count'],
                'mutation_type': cast_astype(mutations['mutation_types'], int),
                'mutation_positions': cast_astype(mutations['positions'], int),
                'path_to_template_circuit': mutations['template_file']
            })
            # Expand the interaction keys in the table
            info_table = update_info_table(info_table, curr_table=mutation_table,
                                           int_stats=interaction_stats,
                                           ref_stats=interaction_stats_og,
                                           result_source_dirs=mutation_dirs,
                                           all_sample_names=all_sample_names,
                                           chosen_sample_names=sample_names,
                                           check_coherent=True)

        if circ_idx != 0 and np.mod(circ_idx, 10) == 0:
            info_table = write_results(info_table)
            info_table = init_info_table()

    write_results(info_table)
    info_table_path = get_pathnames(
        search_dir=data_writer.write_dir, file_key='tabulated_mutation_info.csv', first_only=True)
    info_table = pd.read_csv(info_table_path)
    return info_table
