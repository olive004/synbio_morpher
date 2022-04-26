


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


def get_pathnames(file_key, first_only=False, **kwargs):
    search_folder = os.path.join(*[v for v in kwargs.values()])
    pathnames = glob.glob(os.path.join(search_folder, file_key + '*'))
    if first_only and pathnames:
        logging.info(pathnames)
        pathnames = pathnames[0]
    return pathnames


def get_pathname_by_search_str(folder, search_key):
    file_name = list(map(folder, glob.glob(search_key)))
    return file_name
