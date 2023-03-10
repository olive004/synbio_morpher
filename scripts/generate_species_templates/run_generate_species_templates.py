from functools import partial
import os
from fire import Fire

from src.utils.common.setup_new import construct_circuit_from_cfg, prepare_config
from src.utils.results.experiments import Experiment, Protocol
from src.utils.results.result_writer import ResultWriter
from src.srv.io.manage.script_manager import script_preamble
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.data.fake_data_generation.seq_generator import RNAGenerator
from src.utils.circuit.agnostic_circuits.circuit_manager_new import CircuitModeller


def main(config=None, data_writer=None):
    # set configs
    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
            "scripts", "generate_species_templates", "configs", "base_config.json"))
    config_file = load_json_as_dict(config)
    config_file = prepare_config(config_file)
    exp_configs = config_file.get("circuit_generation")

    # start_experiment
    if data_writer is None:
        data_writer_kwargs = {'purpose': 'generate_species_templates'}
        data_writer = ResultWriter(**data_writer_kwargs)

    protocols = [
        Protocol(
            partial(RNAGenerator(data_writer=data_writer).generate_circuits,
                    iter_count=exp_configs.get("repetitions", 1),
                    count=exp_configs.get("species_count", 3), slength=exp_configs["sequence_length"],
                    proportion_to_mutate=exp_configs.get(
                        "proportion_to_mutate", 0.3),
                    protocol=exp_configs["generator_protocol"]),
            req_output=True,
            name="generating_sequences"
        ),
        [
            Protocol(
                partial(construct_circuit_from_cfg,
                        config_file=config_file),
                req_input=True,
                req_output=True,
                name="making_circuit"
            ),
            Protocol(
                CircuitModeller(
                    result_writer=data_writer, config=config_file).compute_interactions,
                req_input=True,
                name="compute_interaction_strengths"
            )
        ]
    ]
    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
