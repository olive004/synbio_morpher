from copy import deepcopy
from functools import partial
from abc import ABC, abstractmethod
import os
import pandas as pd
from src.utils.data.data_format_tools.common import write_csv, write_json

from src.utils.data.data_format_tools.manipulate_fasta import write_fasta_file
from src.utils.misc.io import create_location, get_subdirectories
from src.utils.misc.string_handling import add_outtype, make_time_str
from src.utils.misc.type_handling import find_sublist_max


class DataWriter():

    def __init__(self, purpose: str, out_location: str = None) -> None:
        self.purpose = purpose
        self.script_dir = os.path.join('scripts')
        self.root_output_dir = os.path.join('data')
        self.exception_dirs = os.path.join('example_data')
        if out_location is None:
            self.original_write_dir = self.make_location(purpose)
        else:
            self.original_write_dir = out_location
        self.write_dir = deepcopy(self.original_write_dir)

    def output(self, out_type: str = None, out_name: str = None, overwrite: bool = False, return_path: bool = False,
               new_file: bool = False, filename_addon: str = None, subfolder: str = None, write_master: bool = True,
               writer=None, **writer_kwargs):
        if self.write_dir in out_name:
            base_name = os.path.basename(out_name)
        if new_file:
            if filename_addon is None:
                filename_addon = make_time_str()
            base_name = f'{out_name}_{filename_addon}'
        else:
            base_name = f'{out_name}'
        if subfolder:
            out_subpath = os.path.join(self.write_dir, subfolder)
            create_location(out_subpath)
            out_path = os.path.join(
                out_subpath, add_outtype(base_name, out_type))
        elif out_type is None:
            out_path = os.path.join(
                self.write_dir, base_name)
        else:
            out_path = os.path.join(
                self.write_dir, add_outtype(base_name, out_type))
        if writer is None:
            writer = self.get_write_func(
                out_type, out_path, overwrite=overwrite)
            writer_kwargs['out_path'] = out_path
        writer(**writer_kwargs)
        if write_master:
            if overwrite and not os.path.exists(out_path):
                self.write_to_master_summary(
                    out_name, out_name=base_name, out_path=out_path,
                    filename_addon=filename_addon, out_type=out_type)
        if return_path:
            return out_path

    def get_write_func(self, out_type: str, out_path: str, overwrite: bool):
        if out_type == "fasta":
            return partial(write_fasta_file, fname=out_path)
        if out_type == "csv":
            return partial(write_csv, out_path=out_path, overwrite=overwrite)
        if out_type == "json":
            return partial(write_json, out_path=out_path, overwrite=overwrite)
        raise ValueError(
            f'No write function available for output of type {out_type}')

    def make_location(self, purpose: str):

        if purpose in get_subdirectories(self.script_dir) or purpose in self.exception_dirs:
            location = os.path.join(self.root_output_dir,
                                    purpose,
                                    self.generate_location_instance())
            create_location(location)
            return location
        raise ValueError(
            f'Unrecognised purpose {purpose} for writing data to.')

    def generate_location_instance(self):
        return make_time_str()

    def subdivide_writing(self, name: str, safe_dir_change=True):
        base_dir = self.original_write_dir if safe_dir_change else self.write_dir
        location = os.path.join(base_dir, name)
        create_location(location)
        self.write_dir = location

    def unsubdivide_last_dir(self):
        self.write_dir = os.path.dirname(self.write_dir)
        if self.original_write_dir not in self.write_dir:
            self.unsubdivide()

    def unsubdivide(self):
        self.write_dir = deepcopy(self.original_write_dir)

    def write_to_master_summary(self, name: str, **kwargs):
        master_summary = {str(k): str(v) for k, v in kwargs.items()}
        master_summary["name"] = name
        self.output('csv', 'master_summary', write_master=False,
                    **{'data': master_summary})


class Tabulated(ABC):

    def __init__(self) -> None:
        self.column_names, self.data = self.get_props_as_split_dict()
        self.max_table_length = find_sublist_max(self.data)

    def as_table(self):
        return pd.DataFrame.from_dict(dict(zip(self.column_names, self.data)))

    @abstractmethod
    def get_props_as_split_dict(self):
        pass
