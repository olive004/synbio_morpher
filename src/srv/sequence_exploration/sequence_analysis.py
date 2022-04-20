

from src.srv.io.results.writer import DataWriter
from src.srv.parameter_prediction.interactions import InteractionMatrix


def analyse_interactions(pathname):

    interactions = InteractionMatrix(matrix_path=pathname)

    stats = interactions.get_stats()

    return stats

    # note sequence mutation method
