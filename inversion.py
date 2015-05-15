
from scipy.optimize import fmin_cg, fmin_l_bfgs_b, fmin_ncg
import numpy as np
import sys
    
class SeisInverseProblem(object):
    
    def __init__(self, sp, c0):
        
        self.sp = sp
        self.x0 = (1./c0.ravel())
        self.arrdims = (self.sp.mesh.nNy, self.sp.mesh.nNx)
        self.freqs = self.sp._subConfigSettings['freqs']
        self._misfits = {}
        self.models = []
        self.gradfac = -1./(len(self.freqs) * self.sp.mesh.nN * sp.nsrc * sp.nrec)

    @staticmethod
    def printNow(txt):
        sys.stdout.write('%s\n'%txt)
        sys.stdout.flush()
    
    def rebuildIfDiff(self, x):
        c = 1./x.reshape(self.arrdims)
        if ((self.sp.systemConfig['c'] - c)**2).sum() > 0.:
            self.sp.rebuildSystem(c)
    
    def f(self, x):
        
        self.rebuildIfDiff(x)
        
        self.sp.forward()
        misfit = self.sp.misfit
        
        fmisfit = np.sqrt(np.sum([misfit[key]**2 for key in misfit]))
        key = hash(x.tostring())
        if not(key in self._misfits):
            self.printNow('\t\t\t%8.2e'%(fmisfit,))
        self._misfits[key] = fmisfit
        self.models.append(1./x.reshape(self.arrdims))
        
        return fmisfit
    
    def g(self, x):
        f = self.f(x)
        self.sp.backprop()
        
        g = self.sp.g
        xg = (x * reduce(np.add, (g[key]*self.freqs[key]**2 for key in g)).ravel()).real.astype(np.float64, order='F') # omega^2/c
        
        return xg * self.gradfac

    def progress(self, xk):
        self.iter += 1
        self.printNow('%3d\t%8.2e'%(self.iter, self._misfits[hash(xk.tostring())])) 

    def __call__(self, solver=None, **kwargs):

        if solver is None:
            solver = fmin_cg

        if 'maxiter' not in kwargs:
            kwargs['maxiter'] = 20
        
        self.printNow('It.\tMisfit\t\tfn Eval.')
        self.iter = 0
        fmisfit = self.f(self.x0)
        self.printNow('  0\t%8.2e'%(fmisfit,))
        res = solver(self.f, self.x0, fprime=self.g, callback=self.progress, **kwargs)
        return res
