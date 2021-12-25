import logging
from fire import Fire
# from scripts.agnostic_simulation.run_agnostic_simulation import main
from scripts.RNA_circuit_simulation.run_RNA_circuit import main

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
  
logger.debug("some debugging...")


if __name__ == "__main__":
    Fire(main)
