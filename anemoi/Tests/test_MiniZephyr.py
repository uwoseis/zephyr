import unittest
import numpy as np
from anemoi import MiniZephyr, SimpleSource

class TestMiniZephyr(unittest.TestCase):

    def setUp(self):
        pass

    def test_forwardModelling(self):

        nx = 100
        nz = 200

        systemConfig = {
            'dx':       1.,                             # m
            'dz':       1.,                             # m
            'c':        2500. * np.ones((nz,nx)),       # m/s
            'rho':      1.    * np.ones((nz,nx)),       # density
            'nx':       nx,                             # count
            'nz':       nz,                             # count
            'freeSurf': [False, False, False, False],   # t r b l
            'nPML':     10,
            'freq':     2e2,
        }

        Ainv = MiniZephyr(systemConfig)
        src = SimpleSource(systemConfig)

        q = src(nx/2, nz/2)
        u = Ainv*q

if __name__ == '__main__':
    unittest.main()