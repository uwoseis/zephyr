'''
Sparse system solvers for Zephyr
'''
from __future__ import unicode_literals, print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import range, object

import types
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

try:
    from pymatsolver import MumpsSolver
    DEFAULT_SOLVER = MumpsSolver
except ImportError:
    DEFAULT_SOLVER = scipy.sparse.linalg.splu

class DirectSolver(object):
    '''
    Wrapper around a direct sparse system solver.
    '''

    def __init__(self, Solver=None):
        '''
        Initialize solver

        Args:
            Solver (class): The class to instantiate as the low-level solver
        '''

        self._Solver = Solver

    @property
    def Solver(self):
        'Returns the solver class of choice'
        if getattr(self, '_Solver', None) is None:
            self._Solver = DEFAULT_SOLVER
        return self._Solver

    @property
    def Ainv(self):
        'Returns a Solver instance'

        if getattr(self, '_Ainv', None) is None:
            self._Ainv = self.Solver(self.A)
        return self._Ainv

    @property
    def A(self):
        'The system matrix'

        if not hasattr(self, '_A'):
            raise Exception('System matrix has not been set')
        return self._A
    @A.setter
    def A(self, A):
        if isinstance(A, scipy.sparse.spmatrix):
            self._A = A
        else:
            raise Exception('Class %s can only register SciPy sparse matrices'%(self.__class__.__name__,))

    @property
    def shape(self):
        return self.A.T.shape

    def __mul__(self, rhs):
        '''
        Carries out the action of solving for wavefields.

        Args:
            rhs (sparse matrix): Right-hand side vector(s)

        Returns:
            np.ndarray: Wavefields
        '''

        if hasattr(self.Ainv, '__mul__'):
            action = lambda b: self.Ainv * b
        elif hasattr(self.Ainv, 'solve'):
            action = lambda b: self.Ainv.solve(b)
        else:
            raise Exception('Can\'t interpret how to use solver class %s'%(self.Ainv.__class__.__name__,))

        if isinstance(rhs, scipy.sparse.spmatrix):
            def qIter(qs):
                for j in range(qs.shape[1]):
                    qi = qs.getcol(j).toarray().ravel()
                    yield qi
                return
        else:
            def qIter(qs):
                for j in range(qs.shape[1]):
                    qi = qs[:,j]
                    yield qi
                return

        result = np.empty(rhs.shape, dtype=np.complex128)
        for i, q in enumerate(qIter(rhs)):
            result[:,i] = action(q)

        return result
