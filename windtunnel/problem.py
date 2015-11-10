
import numpy as np
from anemoi import BaseModelDependent, DiscretizationWrapper, MiniZephyr, Eurus
import SimPEG
from .survey import HelmBaseSurvey, Helm2DSurvey, Helm25DSurvey

try:
    from multiprocessing import Pool, Process
except ImportError:
    PARALLEL = False
else:
    PARALLEL = True

PARTASK_TIMEOUT = 60

class MultiFreq(DiscretizationWrapper):
    
    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'disc':         (True,      '_disc',        None),
        'freqs':        (True,      None,           list), 
        'parallel':     (False,     '_parallel',    bool),
    }
    
    maskKeys = ['disc', 'freqs', 'parallel']
    
    @property
    def parallel(self):
        return PARALLEL and getattr(self, '_parallel', True)
    
    @property
    def spUpdates(self):
        return [{'freq': freq} for freq in self.freqs]
    
    @property
    def disc(self):
        return self._disc

    def __mul__(self, rhs):
        
        if self.parallel:
            pool = Pool()
            plist = []
            for sp in self.subProblems:
                p = pool.apply_async(sp, (rhs,))
                plist.append(p)
            
            u = (self.scaleTerm*p.get(PARTASK_TIMEOUT) for p in plist)
        else:
            u = (self.scaleTerm*sp*rhs for sp in self.subProblems)
        
        return u
    

class HelmBaseProblem(SimPEG.Problem.BaseProblem, BaseModelDependent, DiscretizationWrapper):
    
#    initMap = {
#    #   Argument        Required    Rename as ...   Store as type
#    }

    maskKeys = []
    
    surveyPair = HelmBaseSurvey
    SystemWrapper = MultiFreq
    
    def __init__(self, systemConfig, *args, **kwargs):
        
        # Initialize SimPEG side
        hx = [(self.dx, self.nx-1)]
        hz = [(self.dz, self.nz-1)]
        mesh = SimPEG.Mesh.TensorMesh([hx, hz], '00')
        SimPEG.Problem.BaseProblem.__init__(self, mesh, *args, **kwargs)
        
        # Initialize anemoi side
        DiscretizationWrapper.__init__(self, systemConfig, *args, **kwargs)
    
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
            
    @property
    def system(self):
        if getattr(self, '_system', None) is None:
            self._system = self.SystemWrapper(self.systemConfig)
        return self._system

    @SimPEG.Utils.timeIt
    def Jvec(self, m, v, u=None):
        """Jvec(m, v, u=None)
            Effect of J(m) on a vector v.
            :param numpy.array m: model
            :param numpy.array v: vector to multiply
            :param numpy.array u: fields
            :rtype: numpy.array
            :return: Jv
        """
        raise NotImplementedError('J is not yet implemented.')

    @SimPEG.Utils.timeIt
    def Jtvec(self, m, v, u=None):
        """Jtvec(m, v, u=None)
            Effect of transpose of J(m) on a vector v.
            :param numpy.array m: model
            :param numpy.array v: vector to multiply
            :param numpy.array u: fields
            :rtype: numpy.array
            :return: JTv
        """
        raise NotImplementedError('Jt is not yet implemented.')

    def fields(self, m):
        """
            The field given the model.
            :param numpy.array m: model
            :rtype: numpy.array
            :return: u, the fields
        """
        
        RHSs = self._survey.getRHSorSomething # THIS IS NOT FUNCTIONAL
        field = FancyField([self.system * rhs for rhs in RHSs]) # THIS IS NOT FUNCTIONAL
        return field
        #raise NotImplementedError('fields is not yet implemented.')
    

class Helm2DProblem(HelmBaseProblem):
    
    surveyPair = Helm2DSurvey

    
class Helm25DProblem(HelmBaseProblem):
    
    surveyPair = Helm25DSurvey
