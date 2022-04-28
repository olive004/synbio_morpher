


import glob
import logging
import os


def isolate_filename(filepath: str):
    return os.path.splitext(os.path.basename(filepath))[0]


def get_subdirectories(parent_dir):
    return [name for name in os.listdir(parent_dir)
            if os.path.isdir(os.path.join(parent_dir, name))]


def create_location(pathname):
    if not os.path.isdir(pathname):
        os.umask(0)
        os.makedirs(pathname, mode=0o777)


def get_pathnames(file_key, search_dir, first_only=False):
    path_names = glob.glob(os.path.join(search_dir, '*' + file_key + '*'))
    if first_only and path_names:
        path_names = path_names[0]
    if not path_names:
        raise ValueError(f'Could not find file matching "{file_key}" in {search_dir}.')
    return path_names
