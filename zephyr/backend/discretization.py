
'''
Discretization base classes for Zephyr
'''

import copy
import numpy as np
import scipy.sparse as sp
from .meta import AttributeMapper, BaseSCCache, BaseModelDependent
from .solver import DirectSolver


class BaseDiscretization(BaseModelDependent):
    '''
    Base class for all discretizations.
    '''
    
    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'c':            (True,      '_c',           np.complex128),
        'rho':          (False,     '_rho',         np.float64),
        'freq':         (True,      None,           np.complex128),
        'Solver':       (False,     '_Solver',      None),
        'tau':          (False,     '_tau',         np.float64),
        'premul':       (False,     '_premul',      np.complex128),
    }
    
    @property
    def tau(self):
        'Laplace-domain damping time constant'
        return getattr(self, '_tau', np.inf)
    
    @property
    def dampCoeff(self):
        'Computed damping coefficient to be added to real omega'
        return 1j / self.tau
    
    @property
    def premul(self):
        'A premultiplication factor, used by 2.5D and half differentiation'
        
        return getattr(self, '_premul', 1.)
    
    @property
    def c(self):
        'Complex wave velocity'
        if isinstance(self._c, np.ndarray):
            return self._c
        else:
            return self._c * np.ones((self.nz, self.nx), dtype=np.complex128)
    
    @property
    def rho(self):
        'Bulk density'
        if getattr(self, '_rho', None) is None:
            self._rho = 310. * self.c**0.25
            
        if isinstance(self._rho, np.ndarray):
            return self._rho
        else:
            return self._rho * np.ones((self.nz, self.nx), dtype=np.float64)
    
    @property
    def shape(self):
        return self.A.T.shape
    
    @property
    def Ainv(self):
        'Instance of a Solver class that implements forward modelling'
        
        if not hasattr(self, '_Ainv'):
            self._Ainv = DirectSolver(getattr(self, '_Solver', None))
            self._Ainv.A = self.A.tocsc()
        return self._Ainv
    
    def __mul__(self, rhs):
        'Action of multiplying the inverted system by a right-hand side'
        return (self.Ainv * (self.premul * rhs)).conjugate()
    
    def __call__(self, value):
        return self*value


class DiscretizationWrapper(BaseSCCache):
    '''
    Base class for objects that wrap around discretizations, for example
    in order to model multiple subproblems and distribute configurations
    to different systems.
    '''
    
    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'Disc':         (True,      None,           None),
        'scaleTerm':    (False,     '_scaleTerm',   np.complex128),
    }
    
    maskKeys = {'scaleTerm'}
    
    cacheItems = ['_subProblems']
    
    @property
    def scaleTerm(self):
        'A scaling term to apply to the output wavefield.'
        
        return getattr(self, '_scaleTerm', 1.)
    
    @property
    def _spConfigs(self):
        '''
        Returns subProblem configurations based on the stored
        systemConfig and any subProblem updates.
        '''
        
        def duplicateUpdate(spu):
            nsc = copy.copy(self.systemConfig)
            nsc.update(spu)
            return nsc
        
        return (duplicateUpdate(spu) for spu in self.spUpdates)
    
    @property
    def subProblems(self):
        'Returns subProblem instances based on the discretization.'
        
        if getattr(self, '_subProblems', None) is None:
            
            self._subProblems = map(self.Disc, self._spConfigs)
        return self._subProblems
    
    @property
    def spUpdates(self):
        raise NotImplementedError
    
    def __mul__(self, rhs):
        raise NotImplementedError
