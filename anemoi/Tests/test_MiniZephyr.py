import unittest
import numpy as np
from anemoi import MiniZephyr, SimpleSource, AnalyticalHelmholtz

class TestMiniZephyr(unittest.TestCase):
    
    @staticmethod
    def elementNorm(arr):
        return np.sqrt((arr.conj()*arr).sum()) / arr.size

    def setUp(self):
        pass

    def test_cleanExecution(self):

        systemConfig = {
            'c':        2500.,                          # m/s
            'rho':      1.,                             # density
            'nx':       100,                            # count
            'nz':       200,                            # count
            'freq':     2e2,
        }
        
        xs = 50
        zs = 100

        Ainv = MiniZephyr(systemConfig)
        src = SimpleSource(systemConfig)

        q = src(xs, zs)
        u = Ainv*q
    
    def test_compareAnalytical(self):
        
        systemConfig = {
            'c':        2500.,  # m/s
            'rho':      1.,     # kg/m^3
            'nx':       100,    # count
            'nz':       200,    # count
            'freq':     2e2,    # Hz
        }
        
        xs = 25 
        zs = 25

        Ainv = MiniZephyr(systemConfig)
        src = SimpleSource(systemConfig)
        q = src(xs, zs)
        uMZ = Ainv*q
        
        AH = AnalyticalHelmholtz(systemConfig)
        uAH = AH(xs, zs)
        
        nx = systemConfig['nx']
        nz = systemConfig['nz']
        
        uMZr = uMZ.reshape((nz, nx))
        uAHr = uAH.reshape((nz, nx))
        
        segAHr = uAHr[40:180,40:80]
        segMZr = uMZr[40:180,40:80]
        
        error = self.elementNorm((segAHr - segMZr) / abs(segAHr))
        
        self.assertTrue(error < 1e-2)
    

if __name__ == '__main__':
    unittest.main()