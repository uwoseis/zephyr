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
        #   Argument        Rename to Property
            'disc':         '_discretization',
            'freqs':        None,
            'nky':          '_nky',
            'parallel':     '_parallel',
        }
        
        for key in initMap.keys():
            if key in systemConfig:
                if initMap[key] is None:
                    setattr(self, key, systemConfig[key])
                else:
                    setattr(self, initMap[key], systemConfig[key])
        
        self.systemConfig = {key: systemConfig[key] for key in systemConfig if key not in initMap}
    
    @property
    def discretization(self):
        
        if getattr(self, '_discretization', None) is None:
            from minizephyr import MiniZephyr
            self._discretization = MiniZephyr
        return self._discretization
    
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
            dky = 1. / (self.cmin * (self.nky-1))
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
            return np.min(self.systemConfig['c'])
        else:
            return self._cmin
    
    @property
    def spUpdates(self):
        
        weightfac = 1./(2*self.nky - 1) if self.nky > 1 else 1.
        return [{'freq': freq, 'ky': freq*ky, 'premul': weightfac*(1. + (ky > 0))} for freq in self.freqs for ky in self.pkys]
        
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