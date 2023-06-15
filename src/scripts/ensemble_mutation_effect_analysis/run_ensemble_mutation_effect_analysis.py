

import os
from src.scripts.generate_species_templates.run_generate_species_templates import main as generate_species_templates
from src.scripts.gather_interaction_stats.run_gather_interaction_stats import main as gather_interaction_stats
from src.scripts.mutation_effect_on_interactions_signal.run_mutation_effect_on_interactions_signal import main as mutation_effect_on_interactions_signal
from src.srv.io.manage.script_manager import script_preamble, ensemble_func


def main(config=None, data_writer=None):

    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        # "src", "scripts", "ensemble_mutation_effect_analysis", "configs", "test_ensemble.json"))
        # "src", "scripts", "ensemble_mutation_effect_analysis", "configs", "test_large_scale.json"))
        "src", "scripts", "ensemble_mutation_effect_analysis", "configs", "test_large_scale_loaded.json"))
        # "src", "scripts", "ensemble_mutation_effect_analysis", "configs", "distribution_of_energies.json"))
        # "src", "scripts", "ensemble_mutation_effect_analysis", "configs", "distribution_of_energies_loaded.json"))

    ensemble_func(config, data_writer)
