
import unittest
import numpy as np
from zephyr.backend import AnalyticalHelmholtz, FakeSource

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
        
        sloc = np.array([nx/2, nz/2]).reshape((1,2))
        
        Green = AnalyticalHelmholtz(systemConfig)
        u = Green(sloc)

if __name__ == '__main__':
    unittest.main()
 
