


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


def get_pathnames(file_key, **kwargs):
    import glob
    search_folder = os.path.join(*[v for v in kwargs.values()])
    pathnames = glob.glob(os.path.join(search_folder, file_key + '*'))
    return pathnames
