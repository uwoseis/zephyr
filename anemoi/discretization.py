
import copy
import numpy as np
from .meta import AttributeMapper, BaseSCCache, BaseModelDependent
from .solver import DirectSolver

try:
    from multiprocessing import Pool, Process
except ImportError:
    PARALLEL = False
else:
    PARALLEL = True

PARTASK_TIMEOUT = 60

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

class DiscretizationWrapper(BaseSCCache):
    
    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'disc':         (True,      None,           None),
        'scaleTerm':    (False,     '_scaleTerm',   np.complex128),
    }
    
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


class MultiFreq(DiscretizationWrapper):
    
    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'disc':         (True,      '_disc',        None),
        'freqs':        (True,      None,           list),
        'parallel':     (False,     '_parallel',    bool),
    }
    
    maskKeys = ['disc', 'freqs', 'parallel']
    
    @property
    def parallel(self):
        return PARALLEL and getattr(self, '_parallel', True)
    
    @property
    def spUpdates(self):
        return [{'freq': freq} for freq in self.freqs]
    
    @property
    def disc(self):
        return self._disc

    def __mul__(self, rhs):
        
        if self.parallel:
            pool = Pool()
            plist = []
            for sp in self.subProblems:
                p = pool.apply_async(sp, (rhs,))
                plist.append(p)
            
            u = (self.scaleTerm*p.get(PARTASK_TIMEOUT) for p in plist)
        else:
            u = (self.scaleTerm*(sp*rhs) for sp in self.subProblems)
        
        return u
