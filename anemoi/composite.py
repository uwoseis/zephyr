
import warnings
import numpy as np
import copy

try:
    from multiprocessing import Pool, Process
except ImportError:
    PARALLEL = False
else:
    PARALLEL = True

class Composite25D(object):
    
    def __init__(self, systemConfig):
        
        initMap = {
        #   Argument        Rename as ...   Store as type
            'disc':         ('_disc',       None),
            'nky':          ('_nky',        np.int64),
            'parallel':     ('_parallel',   bool),
            'cmin':         ('_cmin',       np.float64),
            'freq':         (None,          np.complex128),
            'c':            (None,          np.float64),
        }
        
        maskKeys = ['nky', 'disc', 'parallel']
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for key in initMap.keys():
                if key in systemConfig:
                    if initMap[key][1] is None:
                        typer = lambda x: x
                    else:
                        typer = initMap[key][1]
                    if initMap[key][0] is None:
                        setattr(self, key, typer(systemConfig[key]))
                    else:
                        setattr(self, initMap[key][0], typer(systemConfig[key]))
        
        self.systemConfig = {key: systemConfig[key] for key in systemConfig if key not in maskKeys}
    
    @property
    def discretization(self):
        
        if getattr(self, '_disc', None) is None:
            from minizephyr import MiniZephyr
            self._disc = MiniZephyr
        return self._disc
    
    @property
    def nky(self):
        if getattr(self, '_nky', None) is None:
            self._nky = 1
        return self._nky
    
    @property
    def pkys(self):
        
        # By regular sampling strategy
        indices = np.arange(self.nky)
        if self.nky > 1:
            dky = self.freq / (self.cmin * (self.nky-1))
        else:
            dky = 0.
        return indices * dky
    
    @property
    def kyweights(self):
        indices = np.arange(self.nky)
        weights = 1. + (indices > 0)
        return weights
    
    @property
    def cmin(self):
        if getattr(self, '_cmin', None) is None:
            return np.min(self.c)
        else:
            return self._cmin
    
    @property
    def spUpdates(self):
        
        weightfac = 1./(2*self.nky - 1) if self.nky > 1 else 1.
        return [{'ky': ky, 'premul': weightfac*(1. + (ky > 0))} for ky in self.pkys]
        
    @property
    def _spConfigs(self):
        
        def duplicateUpdate(spu):
            nsc = copy.copy(self.systemConfig)
            nsc.update(spu)
            return nsc
        
        return (duplicateUpdate(spu) for spu in self.spUpdates)
    
    @property
    def parallel(self):
        return PARALLEL and getattr(self, '_parallel', True)

    @property
    def subProblems(self):
        if getattr(self, '_subProblems', None) is None:
            
            self._subProblems = map(self.discretization, self._spConfigs)
        return self._subProblems
    
    def __mul__(self, rhs):
        
        if self.parallel:
            pool = Pool()
            plist = []
            for sp in self.subProblems:
                p = pool.apply_async(sp, (rhs,))
                plist.append(p)
            
            u = (p.get(60) for p in plist)
        else:
            u = (sp*rhs for sp in self.subProblems)
        
        return reduce(np.add, u)