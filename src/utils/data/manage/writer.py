from src.utils.data.data_format_tools.manipulate_fasta import write_fasta_file


class DataWriter():

    def __init__(self) -> None:
        pass

    def output(self, data_generator, out_location, out_type, gen_type, gen_run_count):
        writer = self.get_write_func(out_type)
        writer(data_generator, out_location, gen_type, gen_run_count)

    def get_write_func(self, out_type):
        if out_type == "fasta":
            return write_fasta_file
        raise ValueError(f'No write function available for output of type {out_type}')
