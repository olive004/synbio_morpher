

import glob
import os
from typing import List, Union

from src.utils.misc.helper import vanilla_return
from src.utils.misc.string_handling import get_intersecting_string, remove_file_extension


def isolate_filename(filepath: str):
    if type(filepath) == str:
        return os.path.splitext(os.path.basename(filepath))[0]
    return None


def create_location(pathname):
    if not os.path.isdir(pathname):
        os.umask(0)
        os.makedirs(pathname, mode=0o777)


def get_pathnames(search_dir: str, file_key: Union[List, str] = '', first_only: bool = False, allow_empty: bool = False,
                  optional_subdir: str = '', conditional: Union[str, None] = 'filenames'):
    """ Get the pathnames in a folder given a keyword. 

    Args:
    conditional: can be 'filenames' (default) to only return pathnames of files, 
    'directories' to only return pathnames of directories, or None to return all pathnames
    """
    path_condition = vanilla_return
    if conditional == 'directories':
        path_condition = os.path.isdir
    elif conditional == 'filenames':
        path_condition = os.path.isfile

    if type(file_key) == list:
        all_path_names = []
        for fk in file_key:
            all_path_names.append(
                set(sorted([f for f in glob.glob(os.path.join(
                    search_dir, '*' + file_key + '*')) if path_condition(f)]))
            )
        path_names = list(all_path_names[0].intersection(*all_path_names[1:]))
    elif not file_key:
        path_names = sorted([os.path.join(search_dir, f) for f in os.listdir(
            search_dir) if path_condition(os.path.join(search_dir, f))])
    else:
        path_names = sorted([f for f in glob.glob(os.path.join(
            search_dir, '*' + file_key + '*')) if path_condition(f)])
    if first_only and path_names:
        path_names = path_names[0]
    if not path_names and optional_subdir:
        path_names = get_pathnames(os.path.join(search_dir, optional_subdir), file_key=file_key,
                                   first_only=first_only, allow_empty=allow_empty, conditional=conditional)
    if not path_names and not allow_empty:
        raise ValueError(
            f'Could not find file matching "{file_key}" in {search_dir}.')
    return path_names


def get_subdirectories(parent_dir, only_basedir=False):
    # return [name for name in os.listdir(parent_dir)
    #         if os.path.isdir(os.path.join(parent_dir, name))]
    subdirectories = [f.path for f in os.scandir(parent_dir) if f.is_dir()]
    if only_basedir:
        return [os.path.basename(s) for s in subdirectories]
    return subdirectories


def convert_pathname_to_module(filepath: str):
    filepath = remove_file_extension(filepath)
    return os.path.normpath(filepath).replace(os.sep, '.')


def import_module_from_path(module_name: str, filepath: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    return importlib.util.module_from_spec(spec)
