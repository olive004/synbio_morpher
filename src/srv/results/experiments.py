from datetime import datetime
import logging
from typing import List

from src.utils.data.manage.writer import DataWriter


class Protocol():

    def __init__(self, protocol, req_output=False, req_input=False, name='') -> None:
        self.req_output = req_output
        self.req_input = req_input
        self.protocol = protocol
        self.name = name
        self.output = None

    def __call__(self, *input_args):
        if input_args is None:
            output = self.protocol()
        else:
            output = self.protocol(*input_args)
        if self.req_output:
            if output is not None:
                self.output = output
            return output


class Experiment():

    def __init__(self, config: str, protocols: List[Protocol], data_writer: DataWriter) -> None:
        
        self.name = 'experiment'
        self.start_time = datetime.now()
        self.protocols = protocols
        self.total_time = 0
        self.data_writer = data_writer

    def run_experiment(self):
        out = None
        for protocol in self.protocols:
            logging.info(protocol.name)
            if protocol.req_input:
                out = protocol(out)
            else:
                out = protocol()
        self.total_time = datetime.now() - self.start_time
        self.write_experiment()

    def write_experiment(self):
        experiment_data = self.collect_experiment()
        self.data_writer.output(out_type='json', out_name=self.name, data=experiment_data)
        
    def collect_experiment(self):
        return {
            "total_time": str(self.total_time),
            "protocols": [p.name for p in self.protocols],
            "purpose": self.data_writer.purpose
        }
