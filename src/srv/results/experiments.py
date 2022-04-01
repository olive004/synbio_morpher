from datetime import datetime

from src.utils.data.manage.writer import DataWriter


class Experiment():

    def __init__(self, config: str, protocols: list) -> None:
        
        self.start_time = datetime.now()
        self.protocols = protocols
        self.total_time = 0

    def run_experiment(self):
        for protocol in self.protocols:
            protocol()
        self.total_time = datetime.now() - self.start_time
        self.write_experiment()

    def write_experiment(self):
        DataWriter()
        pass
