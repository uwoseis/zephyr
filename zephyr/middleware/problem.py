
import numpy as np
import scipy.sparse as sp
from zephyr.backend import BaseModelDependent, BaseSCCache, MultiFreq, MiniZephyr, Eurus
import SimPEG
from .survey import HelmBaseSurvey, Helm2DSurvey, Helm25DSurvey
from .fields import HelmFields

EPS = 1e-15

class HelmBaseProblem(SimPEG.Problem.BaseProblem, BaseModelDependent, BaseSCCache):
    
#    initMap = {
#    #   Argument        Required    Rename as ...   Store as type
#    }

#    maskKeys = []
    
    surveyPair = HelmBaseSurvey
    SystemWrapper = MultiFreq
    cacheItems = ['_system']
    
    def __init__(self, systemConfig, *args, **kwargs):
         
        # Initialize anemoi side
        BaseSCCache.__init__(self, systemConfig, *args, **kwargs)       
        
        # Initialize SimPEG side
        hx = [(self.dx, self.nx-1)]
        hz = [(self.dz, self.nz-1)]
        mesh = SimPEG.Mesh.TensorMesh([hx, hz], '00')
        SimPEG.Problem.BaseProblem.__init__(self, mesh, *args, **kwargs)

#    @property
#    def _survey(self):
#        return self.__survey
#    @_survey.setter
#    def _survey(self, value):
#        self.__survey = value
#        if value is None:
#            self._cleanSystem()
#        else:
#            self._buildSystem()

    def updateModel(self, m, loneKey='c'):
        
        if m is None:
            return
        
        elif isinstance(m, dict):
            self.systemConfig.update(m)
            self.clearCache()
        
        elif isinstance(m, np.ndarray) or isinstance(m, np.inexact) or isinstance(m, complex) or isinstance(m, float):
            if not np.linalg.norm(m - self.systemConfig.get(loneKey, 0.)) < EPS:
                self.systemConfig[loneKey] = m
                self.clearCache()
        
        else:
                raise Exception('Class %s doesn\'t know how to update with model of type %s'%(self.__class__.__name__, type(m)))
            
    @property
    def system(self):
        if getattr(self, '_system', None) is None:
            self._system = self.SystemWrapper(self.systemConfig)
        return self._system

    @SimPEG.Utils.timeIt
    def Jtvec(self, m=None, v=None, uF=None):
        
        if not self.ispaired:
            raise Exception('%s instance is not paired to a survey'%(self.__class__.__name__,))
            
        if v is None:
            raise Exception('Actually, Jtvec requires a residual vector')
        
        self.updateModel(m)
        
        # v.shape <nrec,  nsrc,  nfreq>
        # o.shape [<nelem, nsrc> . nfreq]
        # r.shape [<nrec, nelem> . nsrc]
                
        resid = v.reshape((self.survey.nrec, self.survey.nsrc, self.survey.nfreq))
        
        if uF is None:
            uF = self._lazyFields(m)
            
        # Make a list of receiver vectors for each frequency, each of size <nelem, nsrc>
        qb = [
              sp.hstack(
               [self.survey.rVec(isrc).T * # <-- <nelem, nrec>
                sp.csc_matrix(resid[:,isrc, ifreq].reshape((self.survey.nrec,1))) # <-- <nrec, 1>
                for isrc in xrange(self.survey.nsrc)
               ] # <-- List comprehension creates sparse vectors of size <nelem, 1> for each source and all receivers
#                (self.survey.rVec(isrc).T * # <-- <nelem, nrec>
#                 sp.csc_matrix(resid[:,isrc, ifreq].reshape((self.survey.nrec,1))) # <-- <nrec, 1>
#                 for isrc in xrange(self.survey.nsrc)
#                ) # <-- Generator expression creates sparse vectors of size <nelem, 1> for each source and all receivers
              ) # <-- Sparse matrix of size <nelem, nsrc> constructed by hstack from generator
              for ifreq in xrange(self.survey.nfreq) # <-- Outer list of size <nfreq>
             ]
        
        uB = self.system * qb
        g = reduce(np.add, ((uFf * uBf).sum(axis=1) for uFf, uBf in zip(uF, uB)))
        
        return g
    
    def _lazyFields(self, m=None):
        
        if not self.ispaired:
            raise Exception('%s instance is not paired to a survey'%(self.__class__.__name__,))
        
        self.updateModel(m)
        
        qs = self.survey.sVecs
        uF = self.system * qs
        
        if not np.iterable(uF):
            uF = [uF]
        
        return uF

    def fields(self, m=None):
        
        uF = self._lazyFields(m)
        fields = HelmFields(self.mesh, self.survey)
        
        for ifreq, uFsub in enumerate(uF):
            fields[:,'u',ifreq] = uFsub
        
        return fields
    

class Helm2DProblem(HelmBaseProblem):
    
    surveyPair = Helm2DSurvey

    
class Helm25DProblem(HelmBaseProblem):
    
    surveyPair = Helm25DSurvey
