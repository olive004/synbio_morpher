


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


def get_pathnames(file_key, root_dir, purpose, experiment_key, subfolder, first_only=False):
    search_folder = os.path.join(root_dir, purpose, experiment_key, subfolder)
    pathnames = glob.glob(os.path.join(search_folder, '*' + file_key + '*'))
    if first_only and pathnames:
        pathnames = pathnames[0]
    return pathnames


def get_pathname_by_search_str(folder, search_key):
    for root, dirs, files in os.walk(folder, topdown=False):
        logging.info(dirs)
        logging.info(files)
    file_name = list(map(folder, glob.glob(search_key)))
    # logging.info(file_name)
    return file_name
