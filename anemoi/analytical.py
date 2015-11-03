
import warnings
import numpy as np
from scipy.special import hankel1

class AnalyticalHelmholtz(object):

    def __init__(self, systemConfig):

        self.omega      = 2 * np.pi * systemConfig['freq']
        self.c          = systemConfig['c']
        self.k          = self.omega / self.c
        self.stretch    = 1. / (1 + (2.*systemConfig.get('eps', 0.)))
        self.theta      = systemConfig.get('theta', 0.)
        self.scaleterm  = systemConfig.get('scaleterm', 0.5)

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
        
        if systemConfig.get('3D', False):
            self.Green = self.Green3D
        else:
            self.Green = self.Green2D

    def Green2D(self, r):

        # Correct: -0.5j * hankel2(0, self.k*r)
        return self.scaleterm * (-0.5j * hankel1(0, self.k*r))
    
    def Green3D(self, r):

        # Correct: (1./(4*np.pi*r)) * np.exp(-1j*self.k*r)
        return self.scaleterm * (1./(4*np.pi*r)) * np.exp(1j*self.k*r)

    def __call__(self, x, z):
        
        dx = self._x - x
        dz = self._z - z
        dist = np.sqrt(dx**2 + dz**2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            strangle = np.arctan(dz / dx) - self.theta
        stretch = np.sqrt(self.stretch * np.cos(strangle)**2 + np.sin(strangle)**2)
        
        return np.nan_to_num(self.Green(dist * stretch)).ravel()
