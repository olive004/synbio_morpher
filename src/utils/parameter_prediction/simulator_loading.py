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

    # path_IntaRNA = "/Users/oliviagallup/Desktop/Kode/Oxford/DPhil/Gene_circuit_glitch_prediction/src/utils/parameter_prediction/IntaRNA"
    # path_IntaRNA = "/Users/oliviagallup/Desktop/Kode/Oxford/DPhil/Gene_circuit_glitch_prediction/src/utils/parameter_prediction/IntaRNA/lib"
    # path_IntaRNA = "/Users/oliviagallup/Desktop/Kode/Oxford/DPhil/Gene_circuit_glitch_prediction/src/utils/parameter_prediction/IntaRNA/lib/libIntaRNA.a"
    # path_IntaRNA = "/Users/oliviagallup/Desktop/Kode/Oxford/DPhil/Gene_circuit_glitch_prediction/src/utils/parameter_prediction/IntaRNA/lib/pkgconfig/IntaRNA.pc"
    # path_IntaRNA = "/Users/oliviagallup/Desktop/Kode/Oxford/DPhil/Gene_circuit_glitch_prediction/src/utils/parameter_prediction/IntaRNA/bin/copomus/IntaRNA.py"
    # path_IntaRNA = "/Users/oliviagallup/Desktop/Kode/Oxford/DPhil/Gene_circuit_glitch_prediction/src/utils/parameter_prediction/IntaRNA/include/IntaRNA"
    # sys.path.insert(0, path_IntaRNA)
    return args