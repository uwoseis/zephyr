
'''
Discretization base classes for Zephyr
'''
from __future__ import division, unicode_literals, print_function, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import map

import copy
from galoshes import BaseSCCache
from problemo import BestSolver as DirectSolver
import numpy as np
import scipy.sparse as sp
from .base import BaseModelDependent


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

        # NB: QC says to merge these two statements. Do not do that. The code
        #     "hasattr(self, '_rho') and not isinstance(self._rho, np.ndarray)"
        #     does not behave the same way in terms of when the 'else' statement
        #     is fired.

        if hasattr(self, '_rho'):
            if not isinstance(self._rho, np.ndarray):
                return self._rho * np.ones((self.nz, self.nx), dtype=np.float64)
        else:
            self._rho = 310. * self.c.real**0.25

        return self._rho

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
    @Ainv.deleter
    def Ainv(self):
        if hasattr(self, '_Ainv'):
            del self._Ainv

    @property
    def factors(self):
        return hasattr(self, '_Ainv')
    @factors.deleter
    def factors(self):
        del self.Ainv

    def __del__(self):
        del self.factors

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

            self._subProblems = list(map(self.Disc, self._spConfigs))
        return self._subProblems

    @property
    def factors(self):
        return not ((not hasattr(self, '_subProblems')) or (not any((sp.factors for sp in self.subProblems))))
    @factors.deleter
    def factors(self):
        if hasattr(self, '_subProblems'):
            for sp in self.subProblems:
                del sp.factors

    @property
    def spUpdates(self):
        raise NotImplementedError

    def __mul__(self, rhs):
        raise NotImplementedError
