
from .meta import BaseSource
import numpy as np

class SimpleSource(BaseSource):
    
    def __init__(self, systemConfig):
        
        super(SimpleSource, self).__init__(systemConfig)
        
        self._z, self._x = np.mgrid[
            self.zorig : self.dz * self.nz : self.dz,
            self.xorig : self.dx * self.nx : self.dx
        ]
    
    def __call__(self, x, z):
        
        dist = np.sqrt((self._x - x)**2 + (self._z - z)**2)
        srcterm = 1.*(dist == dist.min())
        
        return srcterm.ravel() / srcterm.sum()

class StackedSimpleSource(SimpleSource):

    def __call__(self, x, z):

        q = SimpleSource.__call__(self, x, z)
        return np.hstack([q, np.zeros(self._x.size, dtype=np.complex128)])
