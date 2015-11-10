
from .meta import BaseModelDependent
import numpy as np

class BaseSource(BaseModelDependent):
    
    pass


class FakeSource(BaseSource):
    
    def __call__(self, loc):
        
        return loc

    
class SimpleSource(BaseSource):
    
    def __init__(self, systemConfig):
        
        super(BaseSource, self).__init__(systemConfig)
        
        if hasattr(self, 'ny'):
            raise NotImplementedError('Sources not implemented for 3D case')
            self._z, self._y, self._x = np.mgrid[
                self.zorig : self.dz * self.nz : self.dz,
                self.yorig : self.dy * self.ny : self.dy,
                self.xorig : self.dx * self.nx : self.dx
            ]
        else:
            self._z, self._x = np.mgrid[
                self.zorig : self.dz * self.nz : self.dz,
                self.xorig : self.dx * self.nx : self.dx
            ]
    
    def dist(self, loc):
        
        if hasattr(self, 'ny'):
            raise NotImplementedError('Sources not implemented for 3D case')
            dist = np.sqrt((self._x - loc[:,0])**2 + (self._y - loc[:,1])**2 + (self._z - loc[:,2])**2)
        else:
            dist = np.sqrt((self._x - loc[:,0])**2 + (self._z - loc[:,1])**2)
            
        return dist
    
    def __call__(self, loc):
        
        dist = self.dist(loc)
        srcterm = 1.*(dist == dist.min())
        q = srcterm.ravel() / srcterm.sum()
        
        return q
    
    
class StackedSimpleSource(SimpleSource):

    def __call__(self, loc):

        q = super(StackedSimpleSource, self).__call__(loc)
        return np.hstack([q, np.zeros(self._x.size, dtype=np.complex128)])
