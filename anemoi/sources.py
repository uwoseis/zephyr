
import numpy as np

class SimpleSource(object):
    
    def __init__(self, systemConfig):
        
        xorig   = systemConfig.get('xorig', 0.)
        zorig   = systemConfig.get('zorig', 0.)
        dx      = systemConfig.get('dx', 1.)
        dz      = systemConfig.get('dz', 1.)
        nx      = systemConfig['nx']
        nz      = systemConfig['nz']

        self._z, self._x = np.mgrid[
            zorig : dz * nz : dz,
            xorig : dx * nx : dx
        ]
    
    def __call__(self, x, z):
        
        dist = np.sqrt((self._x - x)**2 + (self._z - z)**2)
        srcterm = 1.*(dist == dist.min())
        
        return srcterm.ravel() / srcterm.sum()

class StackedSimpleSource(SimpleSource):

    def __call__(self, x, z):

        q = SimpleSource.__call__(self, x, z)
        return np.hstack([q, np.zeros(self._x.size, dtype=np.complex128)])
