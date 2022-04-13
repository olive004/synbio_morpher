from functools import partial
from abc import ABC, abstractmethod
import os
import pandas as pd
from src.utils.data.data_format_tools.common import write_csv, write_json

from src.utils.data.data_format_tools.manipulate_fasta import write_fasta_file
from src.utils.misc.helper import get_subdirectories
from src.utils.misc.string_handling import add_outtype, make_time_str
from src.utils.misc.type_handling import find_sublist_max


class DataWriter():

    def __init__(self, purpose, out_location=None) -> None:
        self.script_dir = os.path.join('scripts')
        self.root_output_dir = os.path.join('data')
        self.exception_dirs = os.path.join('example_data')
        if out_location is None:
            self.write_dir = self.make_location(purpose)
        else:
            self.write_dir = out_location

    def output(self, out_type, out_name=None, overwrite=False, **writer_kwargs):
        out_path = os.path.join(self.write_dir, add_outtype(out_name, out_type))
        writer = self.get_write_func(out_type, out_path, overwrite=overwrite)
        writer(**writer_kwargs)

    def get_write_func(self, out_type, out_path, overwrite):
        if out_type == "fasta":
            return partial(write_fasta_file, fname=out_path)
        if out_type == "csv":
            return partial(write_csv, out_path=out_path, overwrite=overwrite)
        if out_type == "json":
            return partial(write_json, out_path=out_path, overwrite=overwrite)
        raise ValueError(
            f'No write function available for output of type {out_type}')

    def make_location(self, purpose):

        if purpose in get_subdirectories(self.script_dir) or purpose in self.exception_dirs:
            location = os.path.join(self.root_output_dir,
                                    purpose,
                                    self.generate_location_instance())
            if not os.path.isdir(location):
                os.umask(0)
                os.makedirs(location, mode=0o777)
            return location
        raise ValueError(f'Unrecognised purpose {purpose} for writing data to.')

    def generate_location_instance(self):
        return make_time_str()


class Tabulated(ABC):

    def __init__(self) -> None:
        self.column_names, self.data = self.get_props_as_split_dict()
        self.max_table_length = find_sublist_max(self.data)

    def as_table(self):
        return pd.DataFrame.from_dict(dict(zip(self.column_names, self.data)))

    @abstractmethod
    def get_props_as_split_dict(self):
        pass
