import sys

from src.utils.data.data_format_tools.manipulate_fasta import load_seq_from_FASTA


def find_simulator_loader(simulator: str):
    if simulator == "IntaRNA":
        return load_IntaRNA
    else:
        raise ValueError(f'Simulator "{simulator}" not found.')


def load_IntaRNA(args: dict) -> dict:
    # Currently can only support one query / target sequence
    args['query'] = load_seq_from_FASTA(args['query'])[0]
    args['target'] = load_seq_from_FASTA(args['target'])[0]

    # sys.path.insert(0, path_IntaRNA)
    return args