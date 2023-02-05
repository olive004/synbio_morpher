

import logging
from src.srv.parameter_prediction.simulator import SIMULATOR_UNITS
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.misc.errors import ExperimentError
from src.utils.misc.scripts_io import get_root_experiment_folder, load_experiment_config
from src.utils.misc.units import per_mol_to_per_molecule


INTERACTION_FILE_ADDONS = {
    # 'coupled_binding_rates': SIMULATOR_UNITS['IntaRNA']['rate'],
    'binding_rates_dissociation': SIMULATOR_UNITS['IntaRNA']['rate'],
    'eqconstants': 'eqconstants'
}


def load_param(filepath, param, experiment_config: dict = None) -> dict:
    if experiment_config is None:
        experiment_config = load_experiment_config(
            experiment_folder=get_root_experiment_folder(filepath))
    return per_mol_to_per_molecule(load_json_as_dict(
        experiment_config.get('molecular_params'))[param])


def load_units(filepath, experiment_config: dict = None):

    if experiment_config is None:
        try:
            experiment_config = load_experiment_config(
                experiment_folder=get_root_experiment_folder(filepath))

        except (FileNotFoundError, ExperimentError):
            logging.warning(
                f'Units unknown - supply a valid experiment directory instead of {filepath}')
            return 'unknown'

    simulator_cfgs = experiment_config.get('interaction_simulator')

    if any([i for i in INTERACTION_FILE_ADDONS.keys() if i in filepath]):
        for i, u in INTERACTION_FILE_ADDONS.items():
            if i in filepath:
                return u
    elif simulator_cfgs.get('name') == 'IntaRNA':
        if simulator_cfgs.get('postprocess'):
            return SIMULATOR_UNITS['IntaRNA']['rate']
        else:
            return SIMULATOR_UNITS['IntaRNA']['energy']
    else:
        return 'unknown'
