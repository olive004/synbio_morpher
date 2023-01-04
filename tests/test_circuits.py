import unittest


from tests.shared import five_circuits, mutate, simulate
from tests.shared import config as CONFIG
from src.srv.sequence_exploration.sequence_analysis import load_tabulated_info, b_tabulate_mutation_info


circuits, config, data_writer = five_circuits(CONFIG, data_writer=None)
circuits, config, data_writer = mutate(circuits, config, data_writer)
circuits, config, data_writer = simulate(circuits, config, data_writer)


class TestCircuit(unittest.TestCase):

    def test_interactions(self):
        pass

    def test_refcircuits():
        # make sure that each simulation used the correct reference circuit

        info1 = b_tabulate_mutation_info(data_writer.ensemble_write_dir,
                                        data_writer=data_writer, experiment_config=config)
        source_dirs = [data_writer.ensemble_write_dir]
        info2 = load_tabulated_info(source_dirs)

        assert (info1 == info2).all().all(), f'Loaded info table incorrectly'


def main(args=None):
    t = TestCircuit()
    t.test_interactions()
    unittest.main()


if __name__ == '__main__':
    unittest.main()
