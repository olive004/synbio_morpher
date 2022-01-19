import unittest
from src.utils.system_definition.agnostic_system.base_system import BaseSystem


def main(config_args=None):
    circuit = BaseSystem(config_args)
    circuit.visualise()





class TestBaseSystem(unittest.TestCase):

    def test_sum(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

    def test_sum_tuple(self):
        self.assertEqual(sum((1, 2, 2)), 6, "Should be 6")

    def test

if __name__ == '__main__':
    unittest.main()