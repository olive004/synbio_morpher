from src.utils.data.data_format_tools.common import verify_file_type

def load_seq_from_FASTA(fasta_filename):
    verify_file_type(fasta_filename, 'fasta')

    