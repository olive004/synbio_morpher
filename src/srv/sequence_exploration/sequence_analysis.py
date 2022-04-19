

from src.srv.io.results.writer import DataWriter
from src.srv.parameter_prediction.interactions import InteractionMatrix


def analyse_interactions(pathname):

    interactions = InteractionMatrix(matrix_path=pathname)


    # categorise interactions


def summarise_offline(analytics, out_name, writer: DataWriter):

    writer.output(out_type='csv', out_name=out_name, data=analytics)

    # note sequence mutation method
