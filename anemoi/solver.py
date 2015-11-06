
import types
import numpy as np
import scipy.sparse.linalg

DEFAULT_SOLVER = scipy.sparse.linalg.splu
    
class DirectSolver(object):
    
    def __init__(self, Solver=None):
        
        self._Solver = Solver
    
    @property
    def Solver(self):
        if getattr(self, '_Solver', None) is None:
            self._Solver = DEFAULT_SOLVER
        return self._Solver
    
    @property
    def Ainv(self):
        if getattr(self, '_Ainv', None) is None:
            self._Ainv = self.Solver(self.A)
        return self._Ainv
            
    @property
    def A(self):
        if not hasattr(self, '_A'):
            raise Exception('System matrix has not been set')
        return self._A
    @A.setter
    def A(self, A):
        if isinstance(A, scipy.sparse.csc_matrix) or hasattr(A, 'tocsc'):
            self._A = A
        else:
            raise Exception('Class %s can only register SciPy sparse matrices'%(self.__class__.__name__,))
    
    def __mul__(self, rhs):
        
        if hasattr(self.Ainv, '__mul__'):
            return self.Ainv * rhs
        elif hasattr(self.Ainv, 'solve'):
            return self.Ainv.solve(rhs)
        else:
            raise Exception('Can\'t interpret how to use solver class %s'%(self.Ainv.__class__.__name__,))
