

import os
from scripts.generate_species_templates.run_generate_species_templates import main as generate_species_templates
from scripts.gather_interaction_stats.run_gather_interaction_stats import main as gather_interaction_stats
from scripts.mutation_effect_on_interactions_signal.run_mutation_effect_on_interactions_signal import main as mutation_effect_on_interactions_signal
from src.srv.io.manage.script_manager import Ensembler
from src.srv.io.results.result_writer import ResultWriter
from src.utils.data.data_format_tools.common import load_json_as_dict




def main(config_filepath: str = None, data_writer=None):
    
    if config_filepath is None:
        config_filepath = os.path.join(
            "scripts", "ensemble_mutation_effect_analysis", "configs", "ensemble_mutation_effect_analysis.json")
    config_file = load_json_as_dict(config_filepath)

    if data_writer is None:
        data_writer_kwargs = {'purpose': config_file.get('purpose', 'ensemble_mutation_effect')}
        data_writer = ResultWriter(**data_writer_kwargs)

    ensembler = Ensembler(data_writer=data_writer, subscripts=[
        (generate_species_templates, ),
        (gather_interaction_stats, ),
        (mutation_effect_on_interactions_signal, )
    ])
