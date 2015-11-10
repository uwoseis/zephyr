
import copy
import numpy as np
from .meta import AttributeMapper, BaseModelDependent
from .solver import DirectSolver

class BaseDiscretization(BaseModelDependent):
    
    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'c':            (True,      '_c',           np.complex128),
        'rho':          (False,     '_rho',         np.float64),
        'freq':         (True,      None,           np.complex128),
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
    def Ainv(self):
        if not hasattr(self, '_Ainv'):
            self._Ainv = DirectSolver(getattr(self, '_Solver', None))
            self._Ainv.A = self.A.tocsc()
        return self._Ainv
    
    def __mul__(self, rhs):
        return self.Ainv * rhs
    
    def __call__(self, value):
        return self*value

class DiscretizationWrapper(AttributeMapper):
    
    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'disc':         (True,      None,           None),
        'scaleTerm':    (False,     '_scaleTerm',   np.complex128),
    }
    
    maskKeys = []
    
    def __init__(self, systemConfig):
        
        super(DiscretizationWrapper, self).__init__(systemConfig)
        self.systemConfig = {key: systemConfig[key] for key in systemConfig if key not in self.maskKeys}
    
    @property
    def scaleTerm(self):
        return getattr(self, '_scaleTerm', 1.)
    
    @property
    def _spConfigs(self):
        
        def duplicateUpdate(spu):
            nsc = copy.copy(self.systemConfig)
            nsc.update(spu)
            return nsc
        
        return (duplicateUpdate(spu) for spu in self.spUpdates)
    
    @property
    def subProblems(self):
        if getattr(self, '_subProblems', None) is None:
            
            self._subProblems = map(self.disc, self._spConfigs)
        return self._subProblems
    
    @property
    def spUpdates(self):
        raise NotImplementedError
    
    def __mul__(self, rhs):
        raise NotImplementedError

