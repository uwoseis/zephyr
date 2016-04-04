
import numpy as np
import scipy.sparse as sp
from ..backend import BaseModelDependent, BaseSCCache, MultiFreq, ViscoMultiFreq, ViscoMultiGridMultiFreq
import SimPEG
from .survey import HelmBaseSurvey, Helm2DSurvey, Helm25DSurvey
from .fields import HelmFields

EPS = 1e-15

class HelmBaseProblem(SimPEG.Problem.BaseProblem, BaseModelDependent, BaseSCCache):

    initMap = {
    #   Argument            Required    Rename as ...   Store as type
        'SystemWrapper':    (True,      None,           None),
    }

#    maskKeys = []

    surveyPair = HelmBaseSurvey
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

    def gradientScaler(self, ifreq):

        omega = 2*np.pi * self.survey.freqs[ifreq]
        c = self.system.subProblems[ifreq].c

        return self.survey.postProcessors[ifreq](-(omega**2 / c**3).ravel())

    @SimPEG.Utils.timeIt
    def Jtvec(self, m=None, v=None, u=None):

        if not self.ispaired:
            raise Exception('%s instance is not paired to a survey'%(self.__class__.__name__,))

        if v is None:
            raise Exception('Actually, Jtvec requires a residual vector')

        self.updateModel(m)

        # v.shape <nrec,  nsrc,  nfreq>
        # o.shape [<nelem, nsrc> . nfreq]
        # r.shape [<nrec, nelem> . nsrc]

        resid = v.reshape((self.survey.nrec, self.survey.nsrc, self.survey.nfreq))
        qb = self.survey.getResidualSources(resid)

        mux = (u is None)
        if mux:
            qf = self.survey.getSources()

            if np.isiterable(qb):
                qm = (sp.hstack(qFi, qBi) for qFi, qBi in zip(qf, qb))
                uMux = self.system * qm
            else:
                uMux = self.system * sp.hstack(qf, qb)

            g = reduce(np.add, (self.gradientScaler(ifreq) * pp((uMuxi[:,:self.survey.nsrc] * uMuxi[:,self.survey.nsrc:]).sum(axis=1)) for ifreq, uMuxi, pp in zip(xrange(self.survey.nfreq), uMux, self.survey.postProcessors)))

        else:
            uB = (pp(uBi) for uBi, pp in zip(self.system * qb, self.survey.postProcessors))

            if isinstance(u, HelmFields):
                uIter = (u[:,'u',ifreq] for ifreq in xrange(self.survey.nfreq))
            else:
                uIter = u

            g = reduce(np.add, (self.gradientScaler(ifreq) * (uFi * uBi).sum(axis=1) for ifreq, uFi, uBi in zip(xrange(self.survey.nfreq), uIter, uB))).real

        return g

    def _lazyFields(self, m=None):

        if not self.ispaired:
            raise Exception('%s instance is not paired to a survey'%(self.__class__.__name__,))

        self.updateModel(m)

        qf = self.survey.getSources()
        uF = self.system * qf

        if not np.iterable(uF):
            uF = [uF]

        return uF

    def fields(self, m=None):

        uF = self._lazyFields(m)
        uF = (pp(uFi) for uFi, pp in zip(uF, self.survey.postProcessors))

        fields = HelmFields(self.mesh, self.survey)

        for ifreq, uFsub in enumerate(uF):
            fields[:,'u',ifreq] = uFsub

        return fields


class Helm2DProblem(HelmBaseProblem):

    initMap = {
    #   Argument            Required    Rename as ...   Store as type
        'SystemWrapper':    (False,     None,           None),
    }

    surveyPair = Helm2DSurvey
    SystemWrapper = MultiFreq


class Helm2DViscoProblem(Helm2DProblem):

    SystemWrapper = ViscoMultiFreq


class Helm2DViscoMultiGridProblem(Helm2DProblem):

    SystemWrapper = ViscoMultiGridMultiFreq


class Helm25DProblem(HelmBaseProblem):

    initMap = {
    #   Argument            Required    Rename as ...   Store as type
        'SystemWrapper':    (False,     None,           None),
    }

    surveyPair = Helm25DSurvey
    SystemWrapper = MultiFreq


class Helm25DViscoProblem(Helm25DProblem):

    SystemWrapper = ViscoMultiFreq
