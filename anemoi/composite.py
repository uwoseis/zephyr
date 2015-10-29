import numpy as np
import copy

class Composite25D(object):
    
    def __init__(self, systemConfig):
        
        initMap = {
        #   Argument        Rename to Property
            'disc':         '_discretization',
            'freqs':        None,
            'nky':          '_nky',
            'parallel':     None,
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
        
        return [{'freq': freq, 'ky': freq*ky, 'premul': 1. + (ky > 0)} for freq in self.freqs for ky in self.pkys]
        
    @property
    def _spConfigs(self):
        
        def duplicateUpdate(spu):
            nsc = copy.copy(self.systemConfig)
            nsc.update(spu)
            return nsc
        
        return (duplicateUpdate(spu) for spu in self.spUpdates)
    
    @property
    def parallel(self):
        return getattr(self, '_parallel', False)
    @parallel.setter
    def parallel(self, par):
        if par:
            try:
                import multiprocessing
                self.pool = multiprocessing.Pool()
            except:
                self._parallel = False
            else:
                self._parallel = True
        else:
            self._parallel = False
    
    @property
    def subProblems(self):
        if getattr(self, '_subProblems', None) is None:
            
            self._subProblems = map(self.discretization, self._spConfigs)
        return self._subProblems
    
    def __mul__(self, rhs):
        
        u = (sp * rhs for sp in self.subProblems)
        
        return reduce(np.add, u)