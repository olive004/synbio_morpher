

import pandas as pd


from src.srv.io.results.writer import DataWriter
from src.srv.parameter_prediction.interactions import InteractionMatrix


def generate_interaction_stats(pathname, writer: DataWriter):

    interactions = InteractionMatrix(matrix_path=pathname)

    stats = interactions.get_stats()

    writer.output(out_type='csv', out_name='circuit_stats', data=stats)

    return stats

    # note sequence mutation method


def 
