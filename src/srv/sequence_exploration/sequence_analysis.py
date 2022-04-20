

from src.srv.io.results.writer import DataWriter
from src.srv.parameter_prediction.interactions import InteractionMatrix


def analyse_interactions(pathname):

    interactions = InteractionMatrix(matrix_path=pathname)

    return interactions.get_stats()

    # note sequence mutation method
