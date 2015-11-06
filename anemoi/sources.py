
from .meta import AttributeMapper
import numpy as np

class BaseSource(AttributeMapper):
    
    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'xorig':        (False,     '_xorig',       np.float64),
        'zorig':        (False,     '_zorig',       np.float64),
        'dx':           (False,     '_dx',          np.float64),
        'dz':           (False,     '_dz',          np.float64),
        'nx':           (True,      None,           np.int64),
        'nz':           (True,      None,           np.int64),
    }
    
    @property
    def xorig(self):
        return getattr(self, '_xorig', 0.)

    @property
    def zorig(self):
        return getattr(self, '_zorig', 0.)
    
    @property
    def dx(self):
        return getattr(self, '_dx', 1.)
    
    @property
    def dz(self):
        return getattr(self, '_dz', 1.)


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
