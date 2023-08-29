
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from synbio_morpher.srv.io.loaders.data_loader import DataLoader, GeneCircuitLoader
from synbio_morpher.srv.io.manage.script_manager import script_preamble
from synbio_morpher.srv.parameter_prediction.simulator import SIMULATOR_UNITS
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict
from synbio_morpher.utils.misc.io import get_pathnames, isolate_filename
from synbio_morpher.utils.misc.scripts_io import get_search_dir, load_experiment_config_original
from synbio_morpher.utils.misc.string_handling import prettify_keys_for_label
from synbio_morpher.utils.misc.type_handling import flatten_listlike
from synbio_morpher.utils.parameter_inference.interpolation_grid import make_slice, make_species_interaction_summary, create_parameter_range
from synbio_morpher.utils.results.analytics.naming import get_analytics_types_all, get_true_names_analytics
from synbio_morpher.utils.results.experiments import Experiment, Protocol
from synbio_morpher.utils.results.result_writer import ResultWriter
from synbio_morpher.utils.results.results import ResultCollector
from synbio_morpher.utils.results.visualisation import VisODE


def main(config=None, data_writer: ResultWriter = None):

    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        # 'synbio_morpher', 'scripts', 'parameter_grid_analysis', 'configs', 'heatmap_cfg.json'))
        'synbio_morpher', 'scripts', 'parameter_grid_analysis', 'configs', 'base_config.json'))

    # Load in parameter grid
    config, source_dir = get_search_dir(
        config_searchdir_key='source_parameter_dir', config_file=config)

    def load_parameter_grids():
        all_parameter_grids = {}
        parameter_grids = get_pathnames(source_dir, 'npy')
        for parameter_grid in parameter_grids:
            all_parameter_grids[isolate_filename(
                parameter_grid)] = DataLoader().load_data(parameter_grid)
        return all_parameter_grids

    def get_sample_names(original_config):
        if original_config['data_path']:
            circuit_filepath = original_config['data_path']
            data = GeneCircuitLoader().load_data(circuit_filepath)
            return data.sample_names
        else:
            return sorted(original_config['data'].keys())

    original_config = load_experiment_config_original(
        source_dir, target_purpose='parameter_based_simulation')
    all_parameter_grids = load_parameter_grids()
    shape_parameter_grid = np.shape(
        list(all_parameter_grids.values())[0])
    num_species = shape_parameter_grid[0]
    sample_names = get_sample_names(original_config)
    assert num_species == len(
        sample_names), f'Number of species in parameter grid ({num_species}) does ' \
        f'not equal number of species in original configuration file {sample_names}.'

    # Get relevant configs
    slicing_configs = config['slicing']

    selected_analytics = slicing_configs['analytics']['names']
    if selected_analytics is None:
        selected_analytics = get_true_names_analytics(
            list(all_parameter_grids.keys())) + ['overshoot_asratio']

    def validate_species_cfgs(*cfg_species_lists: list):
        def validate_each(species_name):
            return species_name in sample_names
        for cfg_species_list in cfg_species_lists:
            assert all([validate_each(species_name) for species_name in cfg_species_list]), \
                f'Species {cfg_species_list} from config were not found in list ' \
                f'of species from circuit {sample_names}'

    validate_species_cfgs(flatten_listlike(list(slicing_configs['interactions']['interacting_species'].values())),
                          flatten_listlike(list(slicing_configs['interactions']['non_varying_species_interactions'].values())))

    def run_visualisation(all_parameter_grids, data_writer, selected_analytics,
                          selected_species_interactions, unselected_species_interactions,
                          slicing_configs, sample_names, shape_parameter_grid):

        def visualise_analytic(analytic_name: str, data: np.ndarray,
                               selected_species_interactions: list, original_parameter_range: np.ndarray):
            result_collector = ResultCollector()
            for i, species_name in enumerate(slicing_configs['species_choices']):
                data_per_species = data[i]
                species_interaction_idxs = {
                    species_interaction_summary[k]['species_interaction_idx']: k for k in species_interaction_summary.keys()}
                sorted_species_interactions = [
                    species_interaction_idxs[k] for k in sorted(species_interaction_idxs.keys())]
                ind, cols = list(map(
                    lambda k: original_parameter_range, sorted_species_interactions[:2]))
                data_container = pd.DataFrame(
                    data=np.squeeze(data_per_species),
                    index=ind,
                    columns=cols)
                result_collector.add_result(data_container,
                                            name=analytic_name,
                                            category=None,
                                            vis_func=VisODE(
                                                figsize=(14, 8)).heatmap,
                                            # vis_func=custom_3D_visualisation,
                                            vis_kwargs={'legend': slicing_configs['species_choices'],
                                                        'out_type': 'png',
                                                        # '__setattr__': {'figsize': (10, 10)},
                                                        'xlabel': f'{sorted_species_interactions[0]} interaction strength '\
                                                        f'({SIMULATOR_UNITS["IntaRNA"]["energy"]})',
                                                        'ylabel': f'{sorted_species_interactions[1]} interaction strength '\
                                                        f'({SIMULATOR_UNITS["IntaRNA"]["energy"]})',
                                                        'title': f'{analytic_name.replace("_", " ")} for {sorted_species_interactions[0]} and {sorted_species_interactions[1]}',
                                                        # 'text': {'x': 12, 'y': 0.85, 's': info_text,
                                                        #          'fontsize': 10,
                                                        #          'bbox': dict(boxstyle='round', facecolor='wheat', alpha=1)},
                                                        'vmin': np.min(data_per_species),
                                                        'vmax': np.max(data_per_species)
                                                        # 'figure': {'figsize': (15, 15)}
                                                        })

            data_writer.write_results(result_collector.results, new_report=False,
                                      no_visualisations=False, only_numerical=False,
                                      no_analytics=True, no_numerical=True)

        def visualise_corner(analytic_name: str, data: np.ndarray, species_names: list,
                             species_chosen: list, original_range: np.ndarray, constant_idx: int):

            def collect_minmax(data, idxs, io):
                dmin = []
                dmax = []
                for i, (sii, sij) in enumerate(zip(idxs[0], idxs[1])):
                    for j, (sji, sjj) in enumerate(zip(idxs[0], idxs[1])):
                        if i <= j:
                            continue
                        slices = [(io,)] * len(idxs[0])
                        slices[i] = slice(data.shape[i])
                        slices[j] = slice(data.shape[j])
                        d = np.where(data[tuple(slices)] < np.inf,
                                     data[tuple(slices)], np.nan)
                        dmin.append(np.min(d))
                        dmax.append(np.max(d))
                return dmin, dmax

            idxs = np.triu_indices(len(species_names))
            ticks = [f"{x:.2e}" for x in original_range]

            for s in species_chosen:

                for io, o in enumerate(original_range):

                    dmin, dmax = collect_minmax(data, idxs, io)

                    plt_kwrgs = {
                        'vmin': np.min(dmin),
                        'vmax': np.max(dmax),
                    }

                    fig1, ax1 = plt.subplots(nrows=len(idxs[0]), ncols=len(idxs[0]),
                                             figsize=(7*len(idxs[0]), 6*len(idxs[0])))
                    fig1.subplots_adjust(wspace=0.325, hspace=0.325)
                    fig2, ax2 = plt.subplots(nrows=len(idxs[0]), ncols=len(idxs[0]),
                                             figsize=(7*len(idxs[0]), 6*len(idxs[0])))
                    fig2.subplots_adjust(wspace=0.325, hspace=0.325)
                    for i, (sii, sij) in enumerate(zip(idxs[0], idxs[1])):
                        for j, (sji, sjj) in enumerate(zip(idxs[0], idxs[1])):
                            if i <= j:
                                continue
                            slices = [(io,)] * len(idxs[0])
                            slices[i] = slice(data.shape[i])
                            slices[j] = slice(data.shape[j])

                            d = np.where(data[tuple(slices)] < np.inf,
                                         data[tuple(slices)], np.nan).squeeze()
                            sns.heatmap(d, xticklabels=ticks,
                                        yticklabels=ticks, ax=ax1[i, j])
                            sns.heatmap(
                                d, xticklabels=ticks, yticklabels=ticks, ax=ax2[i, j], **plt_kwrgs)

                            for ax in [ax1[i, j], ax2[i, j]]:
                                ax.set_xlabel(f'{[species_names[sii], species_names[sij]]} '
                                              f'({SIMULATOR_UNITS["IntaRNA"]["energy"]})')
                                ax.set_ylabel(f'{[species_names[sji], species_names[sjj]]} '
                                              f'({SIMULATOR_UNITS["IntaRNA"]["energy"]})')

                    fig1.suptitle(f'{prettify_keys_for_label(analytic_name.split("_wrt")[0])} of {s} for interactions\n(held constant at {o})',
                                  fontsize=14)
                    fig1.savefig(os.path.join(
                        data_writer.write_dir, f'corner_{analytic_name}_{s}_{io}.png'))
                    fig2.suptitle(f'{prettify_keys_for_label(analytic_name.split("_wrt")[0])} of {s} for interactions\n(held constant at {o})',
                                  fontsize=14)
                    fig2.savefig(os.path.join(
                        data_writer.write_dir, f'fullcorner_{analytic_name}_{s}_{io}.png'))
                    plt.clf

        species_interaction_summary = make_species_interaction_summary(
            species_interactions=selected_species_interactions,
            strength_config=slicing_configs['interactions']['strengths'],
            original_config=original_config, sample_names=sample_names)

        slice_indices = make_slice(selected_species_interactions, unselected_species_interactions, slicing_configs,
                                   shape_parameter_grid, sample_names=sample_names, original_config=original_config)
        info_text = '\n'.join(
            ['Held constant:'] + [f'{n}: {v}' for n, v in zip(
                unselected_species_interactions.values(), slicing_configs['interactions']['non_varying_strengths'].values())]
        )
        original_parameter_range = create_parameter_range(
            original_config['parameter_based_simulation'])
        species_names = sorted(set(flatten_listlike(selected_species_interactions.values()) +
                                   flatten_listlike(unselected_species_interactions.values())))
        for analytic_name in selected_analytics:
            data = all_parameter_grids[analytic_name][slice_indices]
            visualise_analytic(analytic_name, data,
                               selected_species_interactions, original_parameter_range)
            visualise_corner(analytic_name, all_parameter_grids[analytic_name][slice_indices[0]],
                             species_names=species_names, species_chosen=slicing_configs[
                                 'species_choices'],
                             original_range=original_parameter_range,
                             constant_idx=config['visualisation']['corner']['constant_idx'])

    experiment = Experiment(config=config, config_file=config, protocols=[
        Protocol(partial(run_visualisation,
                         all_parameter_grids=all_parameter_grids,
                         data_writer=data_writer,
                         selected_analytics=selected_analytics,
                         selected_species_interactions=slicing_configs[
                             'interactions']['interacting_species'],
                         unselected_species_interactions=slicing_configs[
                             'interactions']['non_varying_species_interactions'],
                         slicing_configs=slicing_configs,
                         sample_names=sample_names,
                         shape_parameter_grid=shape_parameter_grid))
    ], data_writer=data_writer)
    experiment.run_experiment()

    return config, data_writer
