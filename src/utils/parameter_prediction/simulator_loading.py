from typing import Dict


def load_IntaRNA(args: Dict):
    args['query'] = load_seq_from_FASTA(args['query'])
