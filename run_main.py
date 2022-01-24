import logging
from fire import Fire
# from scripts.agnostic_simulation.run_agnostic_simulation import main
from scripts.RNA_circuit_simulation.run_RNA_circuit import main

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
FORMAT = "%(filename)s:%(funcName)s():%(lineno)i: %(message)s %(levelname)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


if __name__ == "__main__":
    Fire(main)
