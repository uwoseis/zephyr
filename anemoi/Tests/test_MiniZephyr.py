import unittest
import numpy as np
from anemoi import MiniZephyr, MiniZephyr25D, SimpleSource, AnalyticalHelmholtz

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
        sloc = np.array([xs, zs]).reshape((1,2))

        Ainv = MiniZephyr(systemConfig)
        src = SimpleSource(systemConfig)

        q = src(sloc)
        u = Ainv*q

    def test_cleanExecution25D(self):

        systemConfig = {
            'c':        2500.,                          # m/s
            'rho':      1.,                             # density
            'nx':       100,                            # count
            'nz':       200,                            # count
            'freq':     2e2,
            'nky':      4,
            'parallel': False,
        }
        
        xs = 50
        zs = 100
        sloc = np.array([xs, zs]).reshape((1,2))

        Ainv = MiniZephyr25D(systemConfig)
        src = SimpleSource(systemConfig)

        q = src(sloc)
        u = Ainv*q

    def test_cleanExecution25DParallel(self):

        systemConfig = {
            'c':        2500.,                          # m/s
            'rho':      1.,                             # density
            'nx':       100,                            # count
            'nz':       200,                            # count
            'freq':     2e2,
            'nky':      4,
            'parallel': True,
        }
        
        xs = 50
        zs = 100
        sloc = np.array([xs, zs]).reshape((1,2))

        Ainv = MiniZephyr25D(systemConfig)
        src = SimpleSource(systemConfig)

        q = src(sloc)
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
        sloc = np.array([xs, zs]).reshape((1,2))

        Ainv = MiniZephyr(systemConfig)
        src = SimpleSource(systemConfig)
        q = src(sloc)
        uMZ = Ainv*q
        
        AH = AnalyticalHelmholtz(systemConfig)
        uAH = AH(sloc)
        
        nx = systemConfig['nx']
        nz = systemConfig['nz']
        
        uMZr = uMZ.reshape((nz, nx))
        uAHr = uAH.reshape((nz, nx))
        
        segAHr = uAHr[40:180,40:80]
        segMZr = uMZr[40:180,40:80]
        
        error = self.elementNorm((segAHr - segMZr) / abs(segAHr))
        
        self.assertTrue(error < 1e-2)
        
    def test_compareAnalytical25D(self):
        
        systemConfig = {
            'c':        2500.,  # m/s
            'rho':      1.,     # kg/m^3
            'nx':       100,    # count
            'nz':       200,    # count
            'freq':     2e2,    # Hz
            'nky':      20,
            '3D':       True,
        }
        
        xs = 25 
        zs = 25
        sloc = np.array([xs, zs]).reshape((1,2))

        Ainv = MiniZephyr25D(systemConfig)
        src = SimpleSource(systemConfig)
        q = src(sloc)
        uMZ = Ainv*q
        
        AH = AnalyticalHelmholtz(systemConfig)
        uAH = AH(sloc)
        
        nx = systemConfig['nx']
        nz = systemConfig['nz']
        
        uMZr = uMZ.reshape((nz, nx))
        uAHr = uAH.reshape((nz, nx))
        
        segAHr = uAHr[40:180,40:80]
        segMZr = uMZr[40:180,40:80]
        
        error = self.elementNorm((segAHr - segMZr) / abs(segAHr))
        print(error)
        
        self.assertTrue(error < 1e-2)
    

if __name__ == '__main__':
    unittest.main()
