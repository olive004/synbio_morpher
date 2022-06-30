

from copy import deepcopy
import os
from scripts.generate_species_templates.run_generate_species_templates import main as generate_species_templates
from scripts.gather_interaction_stats.run_gather_interaction_stats import main as gather_interaction_stats
from scripts.mutation_effect_on_interactions_signal.run_mutation_effect_on_interactions_signal import main as mutation_effect_on_interactions_signal
from src.srv.io.manage.script_manager import Ensembler
from src.src.utils.results.result_writer import ResultWriter
from src.utils.data.data_format_tools.common import load_json_as_dict




def main(config=None, data_writer=None):
    
    if config is None:
        config = os.path.join(
            "scripts", "ensemble_mutation_effect_analysis", "configs", "ensemble_mutation_effect_analysis_1.json")
    if type(config) == str:
        config_filepath = config
    else:
        raise ValueError(f'Must load in config as path for config reconfiguration, instead of config as {config}')
    config_file = load_json_as_dict(config)
    ensemble_configs = config_file.get("base_configs_ensemble", {})

    if data_writer is None:
        data_writer_kwargs = {'purpose': config_file.get('purpose', 'ensemble_mutation_effect_analysis')}
        data_writer = ResultWriter(**data_writer_kwargs)

    subscripts = [script for script in ensemble_configs.keys()]
    ensembler = Ensembler(data_writer=data_writer, config_filepath=config_filepath, subscripts=subscripts)
    # [
    #     "generate_species_templates",
    #     "gather_interaction_stats",
    #     "mutation_effect_on_interactions_signal"
    # ])

    ensembler.run()
