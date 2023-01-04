import unittest


from tests.shared import five_circuits, simulate


class TestCircuit(unittest.TestCase):

    def test_interactions(self):
        pass

    def test_refcircuits():
        # make sure that each simulation used the correct reference circuit 
        circuits, config, data_writer = five_circuits()
        five_circuits()


def main(args=None):
    t = TestCircuit()
    t.test_interactions()
    unittest.main()


if __name__ == '__main__':
    unittest.main()
