from src.utils.data.data_format_tools.common import verify_file_type
from Bio import SeqIO


def load_seq_from_FASTA(filename) -> list:
    verify_file_type(filename, 'fasta')
    fasta_records = SeqIO.parse(open(filename),'fasta')
    sequences = []
    for fasta_record in fasta_records:
        sequences.append(str(fasta_record.seq))
    return sequences
