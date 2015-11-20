
import numpy as np
import scipy.sparse as sp
from ..backend import BaseSCCache, SparseKaiserSource
import SimPEG

class HelmSrc(SimPEG.Survey.BaseSrc):
    
    def __init__(self, rxList, loc):
        
        self.loc = loc
        SimPEG.Survey.BaseSrc.__init__(self, rxList)


class HelmRx(SimPEG.Survey.BaseRx):
    
    def __init__(self, locs, rxType=None):
        
        SimPEG.Survey.BaseRx.__init__(self, locs, rxType)


class HelmBaseSurvey(SimPEG.Survey.BaseSurvey, BaseSCCache):

    srcPair = HelmSrc
    
    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'geom':         (True,      None,           dict),
        'freqs':        (True,      None,           tuple),
    }
    
    def __init__(self, *args, **kwargs):
        
        BaseSCCache.__init__(self, *args, **kwargs)
        SimPEG.Survey.BaseSurvey.__init__(self, **kwargs)
        
        if self.mode == 'fixed':
            rxList = HelmRx(self.rLocs)
            rxListGen = lambda sLoc: [rxList]
        elif self.mode == 'relative':
            rxListGen = lambda sLoc: [HelmRx(sLoc + self.rLocs)]
        
        self.srcList = [HelmSrc(rxListGen(loc), loc) for loc in self.sLocs]
    
    @property
    def nfreq(self):
        return len(self.freqs)
    
    @property
    def geom(self):
        return self._geom
    @geom.setter
    def geom(self, value):
        if value.get('mode', 'fixed') not in {'fixed', 'relative'}:
            raise Exception('%s objects only work with \'fixed\' or \'relative\' receiver arrays'%(self.__class__.__name__,))
            
        self._geom = value
    
    @property
    def mode(self):
        return self.geom.get('mode', 'fixed')
    
    @property
    def sLocs(self):
        return self.geom.get('src', None)
    
    @property
    def rLocs(self):
        return self.geom.get('rec', None)
    
    @property
    def sTerms(self):
        return self.geom.get('sterms', np.ones((self.nsrc,), dtype=np.complex128))
    
    @property
    def rTerms(self):
        return self.geom.get('rterms', np.ones((self.nrec,), dtype=np.complex128))
    
    @property
    def nsrc(self):
        try:
            return self.sLocs.shape[0]
        except AttributeError:
            return 0
    
    @property
    def nrec(self):
        try:
            return self.rLocs.shape[0]
        except AttributeError:
            return 0
    
    @property
    def rhsGenerator(self):
        if not hasattr(self, '_rhsGenerator'):
            GeneratorClass = self.geom.get('GeneratorClass', SparseKaiserSource)
            self._rhsGenerator = GeneratorClass(self.systemConfig)
        return self._rhsGenerator
    
    @property
    def sVecs(self):
        if not hasattr(self, '_sVecs'):
            self._sVecs = self.rhsGenerator(self.sLocs) * sp.diags(self.sTerms, 0)
        return self._sVecs
    
    def rVec(self, isrc):
        if self.mode == 'fixed':
            if not hasattr(self, '_rVecs'):
                self._rVecs = (self.rhsGenerator(self.rLocs) * sp.diags(self.rTerms, 0)).T
            return self._rVecs
        
        elif self.mode == 'relative':
            if not hasattr(self, '_rVecs'):
                self._rVecs = {}
            if isrc not in self._rVecs:
                self._rVecs[isrc] = (self.rhsGenerator(self.rLocs + self.sLocs[isrc]) * sp.diags(self.rTerms, 0)).T
            return self._rVecs[isrc]
    
    @property
    def rVecs(self):
        return (self.rVec(i) for i in xrange(self.nsrc))
        
    @property
    def nD(self):
        """Number of data"""
        return self.nsrc * self.nrec * self.nfreq

    @property
    def vnD(self):
        """Vector number of data"""
        return self.nfreq * np.array([src.nD for src in self.srcList])

    @SimPEG.Utils.count
    def projectFields(self, u):
        
        data = np.empty((self.nrec, self.nsrc, self.nfreq), dtype=np.complex128)
        
        for isrc, rVec in enumerate(self.rVecs):
            data[:,isrc,:] = rVec * u[:,isrc,:]
            #for ifreq, freq in enumerate(self.freqs):
            #    data[:,isrc,ifreq] = rVec * u[:,isrc,ifreq]
        
        return data
    
    def _lazyProjectFields(self, u):
        
        data = np.empty((self.nrec, self.nsrc, self.nfreq), dtype=np.complex128)
        
        for ifreq, uFreq in enumerate(u):
            for isrc, rVec in enumerate(self.rVecs):
                data[:,isrc,ifreq] = rVec * uFreq[:,isrc]
        
        return data

    @SimPEG.Utils.count
    @SimPEG.Utils.requires('prob')
    def dpred(self, m=None, u=None):
        
        if u is None:
            u = self.prob._lazyFields(m)
            return self._lazyProjectFields(u).ravel()
        else:
            return self.projectFields(u).ravel()

class Helm2DSurvey(HelmBaseSurvey):
    
    pass

    
class Helm25DSurvey(HelmBaseSurvey):
    
    pass
