import unittest
import numpy as np
from anemoi import AnalyticalHelmholtz

class TestAnalyticalHelmholtz(unittest.TestCase):

    def setUp(self):
        pass

    def test_cleanExecution(self):

        nx = 100
        nz = 200

        systemConfig = {
            'c':        2500.,  # m/s
            'nx':       nx,     # count
            'nz':       nz,     # count
            'freq':     2e2,    # Hz
        }

        Green = AnalyticalHelmholtz(systemConfig)
        u = Green(nx/2, nz/2)

if __name__ == '__main__':
    unittest.main()