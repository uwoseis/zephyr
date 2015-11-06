
from .meta import AttributeMapper
from .solver import DirectSolver
import numpy as np

class BaseDiscretization(AttributeMapper):
    
    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'c':            (True,      '_c',           np.complex128),
        'rho':          (False,     '_rho',         np.float64),
        'freq':         (True,      None,           np.complex128),
        'dx':           (False,     '_dx',          np.float64),
        'dz':           (False,     '_dz',          np.float64),
        'nx':           (True,      None,           np.int64),
        'nz':           (True,      None,           np.int64),
        'freeSurf':     (False,     '_freeSurf',    list),
        'Solver':       (False,     '_Solver',      None),
    }
    
    @property
    def c(self):
        if isinstance(self._c, np.ndarray):
            return self._c
        else:
            return self._c * np.ones((self.nz, self.nx), dtype=np.complex128)
    
    @property
    def rho(self):
        if getattr(self, '_rho', None) is None:
            self._rho = 310. * self.c**0.25 
            
        if isinstance(self._rho, np.ndarray):
            return self._rho
        else:
            return self._rho * np.ones((self.nz, self.nx), dtype=np.float64)
        
    @property
    def dx(self):
        return getattr(self, '_dx', 1.)
    
    @property
    def dz(self):
        return getattr(self, '_dz', self.dx)
    
    @property
    def freeSurf(self):
        if getattr(self, '_freeSurf', None) is None:
            self._freeSurf = (False, False, False, False)
        return self._freeSurf
    
    @property
    def Ainv(self):
        if not hasattr(self, '_Ainv'):
            self._Ainv = DirectSolver(getattr(self, '_Solver', None))
            self._Ainv.A = self.A.tocsc()
        return self._Ainv
    
    def __mul__(self, rhs):
        return self.Ainv * rhs
    
    def __call__(self, value):
        return self*value
