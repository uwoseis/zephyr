import unittest
import numpy as np


class TestZephyr(unittest.TestCase):

    def setUp(self):
        pass # run at the beginning of all tests

    def test_meshDimensions(self):
        self.assertTrue(4, 4)

if __name__ == '__main__':
    unittest.main()
