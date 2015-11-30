
from .discretization import DiscretizationWrapper

try:
    import multiprocessing
except ImportError:
    PARALLEL = False
else:
    PARALLEL = True

PARTASK_TIMEOUT = 60


class BaseMPDist(DiscretizationWrapper):
    
    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'parallel':     (False,     '_parallel',    bool),
        'nWorkers':     (False,     '_nWorkers',    np.int64),
    }
    
    maskKeys = {'parallel'}
    
    @property
    def parallel(self):
        return PARALLEL and getattr(self, '_parallel', True)
    
    @property
    def pool(self):
        
        if self.parallel:
            pool = multiprocessing.Pool(self.nWorkers)
        else:
            raise Exception('Cannot start parallel pool; multiprocessing seems to be unavailable')
    
    @property
    def nWorkers(self):
        return min(getattr(self, '_nWorkers', 100), self.cpuCount)
    
    @property
    def cpuCount(self):
        if self.parallel:
            return multiprocessing.cpu_count()
        else:
            return 1
    
    @property
    def addFields(self):
        
        remCap = self.cpuCount / self.nWorkers
        if (self.nWorkers < self.cpuCount) and remCap > 1:
            
            return {'parallel': True, 'nWorkers': remCap}
        
        else:
            return {}
    
    def __mul__(self, rhs):
        
        if isinstance(rhs, list):
            getRHS = lambda i: rhs[i]
        else:
            getRHS = lambda i: rhs
        
        if self.parallel:
            plist = []
            for i, sub in enumerate(self.subProblems):
                
                p = self.pool.apply_async(sub, (getRHS(i),))
                plist.append(p)
            
            u = (self.scaleTerm*p.get(PARTASK_TIMEOUT) for p in plist)
            
        else:
            u = (self.scaleTerm*(sub*getRHS(i)) for i, sub in enumerate(self.subProblems))
        
        return u


class BaseIPYDist(DiscretizationWrapper):
    
    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'profile':      (False,     '_profile',     str),
    }
    
    maskKeys = {'profile'}
    
    @property
    def profile(self):
        return getattr(self, '_profile', 'default')
    
    @property
    def pClient(self):
        if not hasattr(self, '_pClient'):
            from ipyparallel import Client
            self._pClient = Client(self.profile)
        return self._pClient

    @property
    def dView(self):
        if not hasattr(self, '_dView'):
            self._dView = self.pClient[:]
        return self._dView
    
    @property
    def lView(self):
        if not hasattr(self, '_lView'):
            self._lView = self.pClient.load_balanced_view()
        return self._lView
    
    @property
    def nWorkers(self):
        return len(self.pClient.ids)


class MultiFreq(BaseMPDist):
    
    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'freqs':        (True,      None,           list),
    }
    
    maskKeys = {'freqs'}
    
    @property
    def spUpdates(self):
        vals = []
        for freq in self.freqs:
            vals.append({'freq': freq})
            vals[-1].update(self.addFields)
        return vals
    

class SerialMultiFreq(MultiFreq):
    
    @property
    def parallel(self):
        return False
    
    @property
    def addFields(self):
        return {}
