
import types
import numpy as np
import scipy.sparse
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
        if isinstance(A, scipy.sparse.spmatrix):
            self._A = A
        else:
            raise Exception('Class %s can only register SciPy sparse matrices'%(self.__class__.__name__,))
    
    @property
    def shape(self):
        return self.A.T.shape
    
    def __mul__(self, rhs):
        
        if hasattr(self.Ainv, '__mul__'):
            action = lambda b: self.Ainv * b
        elif hasattr(self.Ainv, 'solve'):
            action = lambda b: self.Ainv.solve(b)
        else:
            raise Exception('Can\'t interpret how to use solver class %s'%(self.Ainv.__class__.__name__,))
        
        if isinstance(rhs, scipy.sparse.spmatrix):
            qIter = lambda qs: (qs.getcol(j).toarray().ravel() for j in xrange(qs.shape[1]))
        else:
            qIter = lambda qs: (qs[:,j] for j in xrange(qs.shape[1]))
        
        result = np.empty(rhs.shape, dtype=np.complex128)
        for i, q in enumerate(qIter(rhs)):
            result[:,i] = action(q)
        
        return result