

import os
from scripts.generate_species_templates.run_generate_species_templates import main as generate_species_templates
from scripts.gather_interaction_stats.run_gather_interaction_stats import main as gather_interaction_stats
from scripts.mutation_effect_on_interactions_signal.run_mutation_effect_on_interactions_signal import main as mutation_effect_on_interactions_signal
from src.srv.io.manage.script_manager import Ensembler, script_preamble
from src.utils.results.result_writer import ResultWriter
from src.utils.data.data_format_tools.common import load_json_as_dict


def main(config=None, data_writer=None):

    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        # "scripts", "ensemble_mutation_effect_analysis", "configs", "test_ensemble.json"))
        "scripts", "ensemble_mutation_effect_analysis", "configs", "test_large_scale.json"))

    config_file = load_json_as_dict(config)

    # Start_experiment
    if data_writer is None:
        data_writer = ResultWriter(
            purpose=config_file.get('experiment', {}).get('purpose'))

    ensembler = Ensembler(data_writer=data_writer, config=config)
    # [
    #     "generate_species_templates",
    #     "gather_interaction_stats",
    #     "mutation_effect_on_interactions_signal"
    # ])

    ensembler.run()
