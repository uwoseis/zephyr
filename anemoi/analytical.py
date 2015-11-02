
import warnings
import numpy as np
from scipy.special import hankel2

class AnalyticalHelmholtz(object):

    def __init__(self, systemConfig):

        self.omega      = 2 * np.pi * systemConfig['freq']
        self.c          = systemConfig['c']
        self.k          = self.omega / self.c
        self.stretch    = 1. / (1 + (2.*systemConfig.get('eps', 0.)))
        self.theta      = systemConfig.get('theta', 0.)

        xorig   = systemConfig.get('xorig', 0.)
        zorig   = systemConfig.get('zorig', 0.)
        dx      = systemConfig.get('dx', 1.)
        dz      = systemConfig.get('dz', 1.)
        nx      = systemConfig['nx']
        nz      = systemConfig['nz']

        self._z, self._x = np.mgrid[
            zorig:zorig+dz*nz:dz,
            xorig:xorig+dz*nx:dx
        ]

    def Green2D(self, r):

        # Correct: -0.5j * hankel2(0, self.k*r)
        return 0.25j * hankel2(0, self.k*r)

    def __call__(self, x, z):
        
        dx = self._x - x
        dz = self._z - z
        dist = np.sqrt(dx**2 + dz**2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            strangle = np.arctan(dz / dx) - self.theta
        stretch = np.sqrt(self.stretch * np.cos(strangle)**2 + np.sin(strangle)**2)
        
        return np.nan_to_num(self.Green2D(dist * stretch)).ravel()
