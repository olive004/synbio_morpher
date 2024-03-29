
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
from datetime import datetime
import logging
from typing import Any, List, Union
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict

from synbio_morpher.utils.results.writer import DataWriter
from synbio_morpher.utils.misc.type_handling import make_attribute_list


class Protocol():

    def __init__(self, protocol_func, req_output=False, req_input=False, name='', skip=False) -> None:
        self.req_output = req_output
        self.req_input = req_input
        self.protocol = protocol_func
        self.name = name
        self.output = None
        self.skip = skip

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

    def __init__(self, config: Union[str, dict], config_file: dict, protocols: List[Protocol], data_writer: DataWriter, debug_inputs=False) -> None:

        self.name = 'experiment'
        self.config = config
        self.config_file = config_file
        self.start_time = datetime.now()
        self.protocols = protocols
        self.total_time = 0
        self.data_writer = data_writer
        self.debug_inputs = debug_inputs
        self.experiment_state = 'uninitiated'

    def run_experiment(self):
        self.experiment_state = 'incomplete'
        self.write_experiment()
        out = None
        self.iterate_protocols(self.protocols, out)

        self.total_time = datetime.now() - self.start_time
        self.experiment_state = 'completed'
        self.write_experiment()

    def iterate_protocols(self, protocols: List[Protocol], out: Any):
        for protocol in protocols:
            if type(protocol) == Protocol:
                if self.debug_inputs:
                    logging.info(f'Input to protocol {protocol.name}: {out}')
                out = self.call_protocol(protocol, out)
                if self.debug_inputs:
                    logging.info(f'Output to protocol {protocol.name}: {out}')
            elif type(protocol) == list and type(out) == list:
                for i, o in enumerate(out):
                    out[i] = self.iterate_protocols(protocols=protocol, out=o)
        return out

    def call_protocol(self, protocol: Protocol, out=None):
        if protocol.req_input and protocol.req_output:
            out = protocol(out)
        elif protocol.req_input:
            protocol(out)
        elif protocol.req_output:
            out = protocol()
        else:
            protocol()
        return out

    def write_experiment(self):
        experiment_data = self.collect_experiment()
        self.data_writer.output(
            out_type='json', out_name=self.name, data=experiment_data, 
            write_master=False, overwrite=True)

    def collect_experiment(self):
        return {
            "total_time": str(self.total_time),
            "purpose": self.data_writer.purpose,
            "config_filepath": self.config,
            "config_params": self.config_file,
            "experiment_state": self.experiment_state
        }
