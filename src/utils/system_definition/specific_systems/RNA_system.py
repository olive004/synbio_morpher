import logging
from src.utils.system_definition.agnostic_system.base_system import BaseSystem


class RNASystem(BaseSystem):
    def __init__(self, config_args=None, simulator="intaRNA"):
        super().__init__(config_args)

        if simulator == "intaRNA":
            from src.utils.parameter_prediction.IntaRNA.bin.CopomuS import CopomuS
            logging.debug("Running CopomuS")
            self.simulator = CopomuS(**config_args)
            self.simulator.main()
    