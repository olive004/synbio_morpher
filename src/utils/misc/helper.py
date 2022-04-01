import os


def next_wrapper(generator):
    return next(generator)


def get_subdirectories(parent_dir):
    return [name for name in os.listdir(parent_dir)
            if os.path.isdir(os.path.join(parent_dir, name))]
