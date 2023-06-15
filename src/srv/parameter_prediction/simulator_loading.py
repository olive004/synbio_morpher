
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
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