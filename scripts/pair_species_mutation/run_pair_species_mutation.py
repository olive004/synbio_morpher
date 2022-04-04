from functools import partial
import os
from fire import Fire
from scripts.common.circuit import construct_circuit_from_cfg

from src.srv.results.experiments import Experiment
from src.utils.data.fake_data_generation.seq_generator import RNAGenerator
from src.utils.evolution.mutation import Evolver


def main():
    # set configs
    config_file = os.path.join(
        "scripts", "pair_species_mutation", "configs", "RNA_pair_species_mutation.json")
    # start_experiment
    data_writer_kwargs = {'purpose': 'pair_species_mutation'}
    protocols = [
        # generate_sequences
        partial(RNAGenerator(**data_writer_kwargs).generate_circuit, count=3, slength=25, protocol="template_mix"
        # partial(RNAGenerator(purpose= 'pair_species_mutation').generate_circuit(), count=3, slength=25, protocol="template_mix"
        # partial(RNAGenerator().generate_circuit(), count=3, slength=25, protocol="template_mix"
                ),
        # generate_mutations
        partial(Evolver),
        # simulate_interactions
        partial(construct_circuit_from_cfg, config_file)
    ]
    experiment = Experiment(config_file, protocols)

    output_visualisations
    write_report


if __name__ == "__main__":
    Fire(main)
