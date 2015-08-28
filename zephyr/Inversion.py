from scipy.optimize import fmin_cg, fmin_l_bfgs_b, fmin_ncg
import numpy as np
import sys

class SeisInverseProblem(object):
    
    def __init__(self, problem, c0):
        
        self.problem = problem 
        self.x0 = (1./c0.ravel())
        self.arrdims = (self.problem.mesh.nNy, self.problem.mesh.nNx)
        self.freqs = self.problem._subConfigSettings['freqs']
        self._misfits = {}
        self.models = []
        self.misfits = []
        self.gradfac = -1./(len(self.freqs) * self.problem.mesh.nN * problem.survey.nD)

    @staticmethod
    def printNow(txt):
        sys.stdout.write('%s\n'%txt)
        sys.stdout.flush()
    
    def rebuildIfDiff(self, x):
        c = 1./x.reshape(self.arrdims)
        if ((self.problem.systemConfig['c'] - c)**2).sum() > 0.:
            self.problem.rebuildSystem(c)

    def _freqScalingTerms(self):

        self.rebuildIfDiff(self.x0)
        self.problem.forward()

        terms = self.problem.misfit
        return {key: terms[key] / terms[0] for key in terms}
    
    def f(self, x):
        
        self.rebuildIfDiff(x)
        
        self.problem.forward()
        misfit = self.problem.misfit
        misfit = self.problem.misfit / self._preScaleFreqs
        
        fmisfit = np.sqrt(np.sum([misfit[key]**2 for key in misfit]))
        key = hash(x.tostring())
        if not(key in self._misfits):
            self.printNow('\t\t\t%8.2e'%(fmisfit,))
        self._misfits[key] = fmisfit
        
        return fmisfit
    
    def g(self, x):
        f = self.f(x)
        self.problem.backprop()
        
        g = self.problem.g
        # stabs = abs(self.problem.remote.e0["endpoint.globalFields['srcTerm']"])
        # stabs = stabs[0] / stabs
        # xg = (x * reduce(np.add, (stabs[key]*g[key]*self.freqs[key]**2 for key in g)).real.ravel()).astype(np.float64, order='F') # omega^2/c
        # xg = (x * reduce(np.add, (g[key]*self.freqs[key]**2 for key in g)).real.ravel()).astype(np.float64, order='F') # omega^2/c
        xg = (x * reduce(np.add, (g[key]*self.freqs[key]**2 for key in (g/self._preScaleFreqs))).real.ravel()).astype(np.float64, order='F') # omega^2/c
        
        return xg * self.gradfac

    def progress(self, xk):
        self.iter += 1
        misfit = self._misfits[hash(xk.tostring())]
        self.printNow('%3d\t%8.2e'%(self.iter, misfit))
        self.misfits.append(misfit)
        self.models.append(1./xk.reshape(self.arrdims))

    def __call__(self, solver=None, **kwargs):

        if solver is None:
            solver = fmin_cg

        if 'maxiter' not in kwargs:
            kwargs['maxiter'] = 20

        if kwargs.pop('freqScale', False):
            self._preScaleFreqs = self._freqScalingTerms()
        else:
            self._preScaleFreqs = {key: 1. for key in xrange(len(self.freqs))}
        
        self.printNow('It.\tMisfit\t\tfn Eval.')
        self.iter = 0
        fmisfit = self.f(self.x0)
        self.printNow('  0\t%8.2e'%(fmisfit,))
        res = solver(self.f, self.x0, fprime=self.g, callback=self.progress, **kwargs)
        return res

def doInversion(problem, x0, solverChoice='lbfgs', maxiter=10, outfile='result.hdf5'):

    import hickle

    solvermap = {
        'lbfgs':    fmin_l_bfgs_b,
        'ncg':      fmin_ncg,
        'cg':       fmin_cg,
    }

    solver = solvermap.get(solverChoice, fmin_l_bfgs_b)

    sim = SeisInverseProblem(problem, x0)
    res = sim(solver=solver, maxiter=maxiter)

    final = 1./(res[0] if type(res) is tuple else res).reshape(sim.arrdims)

    resStructure = {
        'initial_model':    x0,
        'solver':           solverChoice,
        'misfits':          sim.misfits,
        'models':           sim.models,
        'final_model':      final,
    }

    if type(res) is tuple:
        resStructure.update({
            'final_misfit': res[1],
            'status':       res[2],
        })

    hickle.dump(resStructure, outfile, compression='gzip')

    return sim
    