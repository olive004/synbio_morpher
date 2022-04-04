import os
from src.utils.data.data_format_tools.manipulate_fasta import write_fasta_file
from src.utils.misc.helper import get_subdirectories
from src.utils.misc.string_handling import make_time_str


class DataWriter():

    def __init__(self, purpose, out_location=None) -> None:
        self.script_dir = os.path.join('scripts')
        self.root_output_dir = os.path.join('data')
        self.exception_dirs = os.path.join('example_data')
        if out_location is None:
            self.write_dir = self.make_location(purpose)
        else:
            self.write_dir = out_location

    def output(self, data_generator, out_location, out_type, gen_type, gen_run_count):
        writer = self.get_write_func(out_type)
        writer(data_generator, out_location, gen_type, gen_run_count)

    def get_write_func(self, out_type):
        if out_type == "fasta":
            return write_fasta_file
        raise ValueError(
            f'No write function available for output of type {out_type}')

    def make_location(self, purpose):

        if purpose in get_subdirectories(self.script_dir) or purpose in self.exception_dirs:
            return os.path.join(self.root_output_dir,
                                purpose,
                                self.generate_location_instance())
        raise ValueError(f'Unrecognised purpose for writing data to {purpose}')

    def generate_location_instance(self):
        return make_time_str()
