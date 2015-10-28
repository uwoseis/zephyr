
import numpy as np
from scipy.special import hankel2

class AnalyticalHelmholtz(object):

    def __init__(self, sc):

        self.omega      = 2 * np.pi * sc['freq']
        self.c          = sc['c']
        self.k          = self.omega / self.c
        self.stretch    = 1. + 2.*sc.get('eps', 0.)
        self.theta      = sc.get('theta', 0.)
        self.xstretch   = np.sqrt(np.sin(self.theta)**2 + self.stretch * np.cos(self.theta)**2)
        self.zstretch   = np.sqrt(np.cos(self.theta)**2 + self.stretch * np.sin(self.theta)**2)

        xorig   = sc.get('xorig', 0.)
        zorig   = sc.get('zorig', 0.)
        dx      = sc.get('dx', 1.)
        dz      = sc.get('dz', 1.)
        nx = sc['nx']
        nz = sc['nz']

        self._z, self._x = np.mgrid[
            zorig:zorig+dz*nz:dz,
            xorig:xorig+dz*nx:dx
        ]

    def Green2D(self, r):

        # Correct: -0.5j * hankel2(0, self.k*r)
        return 0.25j * hankel2(0, self.k*r)

    def __call__(self, x, z):

        return np.nan_to_num(self.Green2D(np.sqrt((self.xstretch * (x - self._x))**2 + (self.zstretch * (z - self._z))**2))).ravel()
