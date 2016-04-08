import unittest
import numpy as np
from zephyr.backend import Eurus, StackedSimpleSource, AnalyticalHelmholtz

class TestEurus(unittest.TestCase):

    @staticmethod
    def _elementNorm(arr):
        return np.sqrt((arr.conj()*arr).sum()) / arr.size

    def setUp(self):
        pass

    def test_cleanExecution(self):

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

        Ainv = Eurus(systemConfig)
        src = StackedSimpleSource(systemConfig)

        sloc = np.array([nx/2, nz/2]).reshape((1,2))
        q = src(sloc)
        u = Ainv*q

    def test_compareAnalytical_Isotropic(self):

        dx = 1
        dz = 1
        nx = 100
        nz = 200
        velocity    = 2000.     * np.ones((nz,nx))
        density     = 1.        * np.ones((nz,nx))

        # Anisotropy parameters
        theta       = 0.        * np.ones((nz,nx))
        epsilon     = 0.        * np.ones((nz,nx))
        delta       = 0.        * np.ones((nz,nx))
        nPML        = 10
        freeSurf    = [False, False, False, False]

        systemConfig = {
            'c':        velocity,  # m/s
            'rho':      density,     # kg/m^3
            'freq':     2e2,    # Hz
            'nx':       nx,
            'nz':       nz,
            'dx':       dx,
            'dz':       dz,
            'theta':    theta,
            'eps':      epsilon,
            'delta':    delta,
            'nPML':     nPML,
            'cPML':     1e3,
            'freeSurf': freeSurf,
        }

        xs = 25
        zs = 25
        sloc = np.array([xs, zs]).reshape((1,2))

        Ainv = Eurus(systemConfig)
        src = StackedSimpleSource(systemConfig)
        q = src(sloc)
        uMZ = Ainv*q

        AH = AnalyticalHelmholtz(systemConfig)
        uAH = AH(sloc)

        nx = systemConfig['nx']
        nz = systemConfig['nz']

        uMZr = uMZ[:nx*nz].reshape((nz, nx))
        uAHr = uAH.reshape((nz, nx))

        segAHr = uAHr[40:180,40:80]
        segMZr = uMZr[40:180,40:80]

        error = self._elementNorm((segAHr - segMZr) / abs(segAHr))

        self.assertTrue(error < 3e-2)

    def test_compareAnalytical_Elliptical(self):

        dx = 1
        dz = 1
        nx = 100
        nz = 200
        velocity    = 2000.     * np.ones((nz,nx))
        density     = 1.        * np.ones((nz,nx))

        # Anisotropy parameters
        theta       = 0.        * np.ones((nz,nx))
        epsilon     = 0.2        * np.ones((nz,nx))
        delta       = 0.2        * np.ones((nz,nx))
        nPML        = 10
        freeSurf    = [False, False, False, False]

        systemConfig = {
            'c':        velocity,  # m/s
            'rho':      density,     # kg/m^3
            'freq':     2e2,    # Hz
            'nx':       nx,
            'nz':       nz,
            'dx':       dx,
            'dz':       dz,
            'theta':    theta,
            'eps':      epsilon,
            'delta':    delta,
            'nPML':     nPML,
            'cPML':     1e3,
            'freeSurf': freeSurf,
        }

        xs = 25
        zs = 25
        sloc = np.array([xs, zs]).reshape((1,2))

        Ainv = Eurus(systemConfig)
        src = StackedSimpleSource(systemConfig)
        q = src(sloc)
        uMZ = Ainv*q

        AH = AnalyticalHelmholtz(systemConfig)
        uAH = AH(sloc)

        nx = systemConfig['nx']
        nz = systemConfig['nz']

        uMZr = uMZ[:nx*nz].reshape((nz, nx))
        uAHr = uAH.reshape((nz, nx))

        segAHr = uAHr[40:180,40:80]
        segMZr = uMZr[40:180,40:80]

        error = self._elementNorm((segAHr - segMZr) / abs(segAHr))

        self.assertTrue(error < 3e-2)

if __name__ == '__main__':
    unittest.main()
