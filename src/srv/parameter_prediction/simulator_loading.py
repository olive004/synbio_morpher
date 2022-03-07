from src.utils.data.data_format_tools.manipulate_fasta import load_seq_from_FASTA


extra_simulators = [
    "CopomuS"
]


def find_simulator_loader(simulator: str):
    if simulator == "IntaRNA":
        return load_IntaRNA
    else:
        raise ValueError(f'Simulator "{simulator}" not found.')


def load_IntaRNA(args: dict) -> dict:
    if args.get('query', None) is None and args.get('target', None) is None: 
        args['query'] = None
        args['target'] = None
    else:
        args['query'] = load_seq_from_FASTA(args['query'])[0]
        args['target'] = load_seq_from_FASTA(args['target'])[0]

    return args