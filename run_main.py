import logging
from fire import Fire
# from scripts.agnostic_simulation.run_agnostic_simulation import main
# from scripts.RNA_circuit_simulation.run_RNA_circuit import main
# from src.utils.data.fake_data_generation.nc_sequences import main
# from scripts.pair_species_mutation.run_pair_species_mutation import main
# from scripts.generate_species_templates.run_generate_species_templates import main
# from scripts.gather_interaction_stats.run_gather_interaction_stats import main
# from scripts.mutation_effect_on_interactions.run_mutation_effect_on_interactions import main
# from scripts.analyse_mutated_templates.run_analyse_mutated_templates import main
from scripts.intarna_sandbox.run_intarna_sandbox import main

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
FORMAT = "%(filename)s:%(funcName)s():%(lineno)i: %(message)s %(levelname)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    Fire(main)
