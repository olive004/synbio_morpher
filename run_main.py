import logging
from fire import Fire
# from scripts.agnostic_simulation.run_agnostic_simulation import main
from scripts.RNA_circuit_simulation.run_RNA_circuit import main

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


if __name__ == "__main__":
    Fire(main)
