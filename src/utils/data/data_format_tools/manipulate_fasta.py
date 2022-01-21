from sympy import sequence
# from src.utils.data.data_format_tools.common import verify_file_type
from Bio import SeqIO
from typing import Union, Dict, List


def load_seq_from_FASTA(filename, as_type="dict") -> Union[Dict,List]:
    # verify_file_type(filename, 'fasta')
    fasta_records = SeqIO.parse(open(filename),'fasta')
    if as_type=="dict":
        sequences = {}
        for fasta_record in fasta_records:
            sequences[fasta_record.id] = str(fasta_record.seq)
        return sequences
    else:
        sequences = []
        for fasta_record in fasta_records:
            sequences.append(str(fasta_record.seq))
        return sequences

load_seq_from_FASTA("./src/utils/data/example_data/toy_mRNA_circuit.fasta")