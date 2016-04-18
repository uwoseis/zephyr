
from galoshes import BaseSCCache
import numpy as np
import scipy.sparse as sp
from ..backend import SparseKaiserSource, MultiGridHelper
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
        'sterms':       (False,     '_sterms',      np.complex128),
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
        return self.geom.get('src')

    @property
    def rLocs(self):
        return self.geom.get('rec')

    @property
    def ssTerms(self):
        return self.geom.get('sterms', np.ones((self.nsrc,), dtype=np.complex128))

    @property
    def srTerms(self):
        return self.geom.get('rterms', np.ones((self.nrec,), dtype=np.complex128))

    @property
    def tsTerms(self):
        return getattr(self, '_sterms', np.ones(self.nfreq, dtype=np.complex128))

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
    def RHSGenerator(self):
        if not hasattr(self, '_RHSGenerator'):
            self._RHSGenerator = self.geom.get('GeneratorClass', SparseKaiserSource)
        return self._RHSGenerator

    def sVecs(self):
        if not hasattr(self, '_sVecs'):
            self._sVecs = self.RHSGenerator(self.systemConfig)(self.sLocs) * sp.diags((self.ssTerms,), (0,))
        return self._sVecs

    def rVec(self, isrc):
        if self.mode == 'fixed':
            if not hasattr(self, '_rVecs'):
                self._rVecs = (self.RHSGenerator(self.systemConfig)(self.rLocs) * sp.diags((self.srTerms,), (0,))).T
            return self._rVecs

        elif self.mode == 'relative':
            if not hasattr(self, '_rVecs'):
                self._rVecs = {}
            if isrc not in self._rVecs:
                self._rVecs[isrc] = (self.RHSGenerator(self.systemConfig)(self.rLocs + self.sLocs[isrc]) * sp.diags((self.srTerms,), (0,))).T
            return self._rVecs[isrc]

    def rVecs(self, ifreq):
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

        for isrc, src in enumerate(self.srcList):
            data[:,isrc,:] = self.rVec(isrc) * u[src,'u',:]
            #for ifreq, freq in enumerate(self.freqs):
            #    data[:,isrc,ifreq] = rVec * u[:,isrc,ifreq]

        return data

    def _lazyProjectFields(self, u):

        data = np.empty((self.nrec, self.nsrc, self.nfreq), dtype=np.complex128)

        for ifreq, uFreq in enumerate(u):
            for isrc, rVec in enumerate(self.rVecs(ifreq)):
                data[:,isrc,ifreq] = rVec * uFreq[:,isrc]

        return data

    def getSources(self):
        qs = self.sVecs()
        if isinstance(self.tsTerms, list) or isinstance(self.tsTerms, np.ndarray):
            qs = [qs * sp.diags((sterm.conjugate(),),(0,)) for sterm in self.tsTerms]

        return qs

    def getResidualSources(self, resid):

        # Make a list of receiver vectors for each frequency, each of size <nelem, nsrc>
        qb = [
              sp.hstack(
               [self.rVec(isrc).T * # <-- <nelem, nrec>
                sp.csc_matrix(resid[:,isrc, ifreq].reshape((self.nrec,1))) # <-- <nrec, 1>
                for isrc in xrange(self.nsrc)
               ] # <-- List comprehension creates sparse vectors of size <nelem, 1> for each source and all receivers
#                (self.rVec(isrc).T * # <-- <nelem, nrec>
#                 sp.csc_matrix(resid[:,isrc, ifreq].reshape((self.survey.nrec,1))) # <-- <nrec, 1>
#                 for isrc in xrange(self.nsrc)
#                ) # <-- Generator expression creates sparse vectors of size <nelem, 1> for each source and all receivers
              ) # <-- Sparse matrix of size <nelem, nsrc> constructed by hstack from generator
              for ifreq in xrange(self.nfreq) # <-- Outer list of size <nfreq>
             ]

        return qb

    @SimPEG.Utils.count
    @SimPEG.Utils.requires('prob')
    def dpred(self, m=None, u=None):

        if u is None:
            u = self.prob.lazyFields(m)
            return self._lazyProjectFields(u).ravel()
        else:
            return self.projectFields(u).ravel()

    @property
    def postProcessors(self):
        return [lambda x: x for _ in self.freqs]

    @property
    def preProcessors(self):
        return [lambda x: x for _ in self.freqs]


class HelmMultiGridSurvey(HelmBaseSurvey):

    @property
    def mgHelper(self):
        'MultiGridHelper instance'

        if not hasattr(self, '_mgHelper'):
            self._mgHelper = MultiGridHelper(self.systemConfig)
        return self._mgHelper

    @property
    def postProcessors(self):
        return self.mgHelper.upScalers

    @property
    def preProcessors(self):
        return self.mgHelper.downScalers

    @property
    def RHSGenerator(self):
        if not hasattr(self, '_RHSGenerator'):
            self._RHSGenerator = self.geom.get('GeneratorClass', SparseKaiserSource)
        return self._RHSGenerator

    @property
    def scScales(self):
        if not hasattr(self, '_scScales'):
            self._scScales = {}
        return self._scScales

    def buildSC(self, ifreq):

        hs = hash(self.mgHelper.scales[ifreq])
        if not hs in self.scScales:
            sc = {key: self.systemConfig[key] for key in self.systemConfig}
            sc.update(self.mgHelper.downScalers[ifreq].scaleUpdate)
            self.scScales[hs] = sc
        return hs

    def sVecs(self, ifreq):

        hs = self.buildSC(ifreq)
        sc = self.scScales[hs]

        return self.RHSGenerator(sc)(self.sLocs) * sp.diags((self.ssTerms), (0,))

    def rVec(self, isrc, ifreq):

        hs = self.buildSC(ifreq)

        if not hasattr(self, '_rVecs'):
            self._rVecs = {}

        if self.mode == 'fixed':
            if hs not in self._rVecs:
                sc = self.scScales[hs]
                self._rVecs[hs] = (self.RHSGenerator(sc)(self.rLocs) * sp.diags((self.srTerms,), (0,))).T
            return self._rVecs[hs]

        elif self.mode == 'relative':
            if hs not in self._rVecs:
                self._rVecs[hs] = {}
            if isrc not in self._rVecs:
                sc = self.scScales[hs]
                self._rVecs[hs][isrc] = (self.RHSGenerator(sc)(self.rLocs + self.sLocs[isrc]) * sp.diags((self.srTerms,), (0,))).T
            return self._rVecs[isrc]

    def rVecs(self, ifreq):
        return (self.rVec(i, ifreq) for i in xrange(self.nsrc))

    @SimPEG.Utils.count
    def projectFields(self, u):

        data = np.empty((self.nrec, self.nsrc, self.nfreq), dtype=np.complex128)

        for isrc, src in enumerate(self.srcList):
            for ifreq in xrange(self.nfreq):
                data[:,isrc,ifreq] = self.rVec(isrc, ifreq) * (self.mgHelper.downScalers[ifreq] * u[src,'u',ifreq]).ravel()
            #for ifreq, freq in enumerate(self.freqs):
            #    data[:,isrc,ifreq] = rVec * u[:,isrc,ifreq]

        return data

    def _lazyProjectFields(self, u):

        data = np.empty((self.nrec, self.nsrc, self.nfreq), dtype=np.complex128)

        for ifreq, uFreq in enumerate(u):
            for isrc, rVec in enumerate(self.rVecs(ifreq)):
                data[:,isrc,ifreq] = rVec * uFreq[:,isrc]

        return data


    def getSources(self):

        if isinstance(self.tsTerms, list) or isinstance(self.tsTerms, np.ndarray):
            qs = [self.sVecs(ifreq) * sp.diags((sterm.conjugate(),), (0,)) if np.iterable(sterm) else sterm.conjugate() * self.sVecs(ifreq) for ifreq, sterm in enumerate(self.tsTerms)]
        else:
            sterm = self.tsTerms
            qs = [sterm.conjugate() * self.sVecs(ifreq) for ifreq in xrange(self.nfreq)]

        return qs

    def getResidualSources(self, resid):

        # Make a list of receiver vectors for each frequency, each of size <nelem, nsrc>
        qb = [
              sp.hstack(
               [self.rVec(isrc, ifreq).T * # <-- <nelem, nrec>
                sp.csc_matrix(resid[:,isrc, ifreq].reshape((self.nrec,1))) # <-- <nrec, 1>
                for isrc in xrange(self.nsrc)
               ] # <-- List comprehension creates sparse vectors of size <nelem, 1> for each source and all receivers
#                (self.rVec(isrc).T * # <-- <nelem, nrec>
#                 sp.csc_matrix(resid[:,isrc, ifreq].reshape((self.survey.nrec,1))) # <-- <nrec, 1>
#                 for isrc in xrange(self.nsrc)
#                ) # <-- Generator expression creates sparse vectors of size <nelem, 1> for each source and all receivers
              ) # <-- Sparse matrix of size <nelem, nsrc> constructed by hstack from generator
              for ifreq in xrange(self.nfreq) # <-- Outer list of size <nfreq>
             ]

        return qb


class Helm2DSurvey(HelmBaseSurvey):

    pass


class Helm2DMultiGridSurvey(Helm2DSurvey, HelmMultiGridSurvey):

    pass


class Helm25DSurvey(HelmBaseSurvey):

    pass


class Helm25DMultiGridSurvey(Helm25DSurvey, HelmMultiGridSurvey):

    pass
