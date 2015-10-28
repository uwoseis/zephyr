import unittest
import numpy as np
from anemoi import AnalyticalHelmholtz

class TestMiniZephyr(unittest.TestCase):

    def setUp(self):
        pass

    def test_forwardModelling(self):

        nx = 100
        nz = 200

        systemConfig = {
            'dx':       1.,     # m
            'dz':       1.,     # m
            'c':        2500.,  # m/s
            'nx':       nx,     # count
            'nz':       nz,     # count
            'freq':     2e2,    # Hz
        }

        Green = AnalyticalHelmholtz(systemConfig)
        u = Green(nx/2, nz/2)

if __name__ == '__main__':
    unittest.main()