import numpy as np
import scipy.sparse as sp
import SimPEG

DEFAULT_FREESURF_BOUNDS = [False, False, False, False]
DEFAULT_IREG = 4
HC_KAISER = {
    1:  1.24,
    2:  2.94,
    3:  4.53,
    4:  6.31,
    5:  7.91,
    6:  9.42,
    7:  10.95,
    8:  12.53,
    9:  14.09,
    10: 14.18,
}

def KaiserWindowedSinc(ireg, offset):
    '''
    Finds 2D source terms to approximate a band-limited point source, based on

    Hicks, Graham J. (2002) Arbitrary source and receiver positioning in finite-difference
        schemes using Kaiser windowed sinc functions. Geophysics (67) 1, 156-166.

    KaiserWindowedSinc(ireg, offset) --> 2D ndarray of size (2*ireg+1, 2*ireg+1)
    Input offset is the 2D offsets in fractional gridpoints between the source location and
    the nearest node on the modelling grid.
    '''
     
    from scipy.special import i0 as bessi0
    import warnings

    try:
        b = HC_KAISER.get(ireg)
    except KeyError:
        print('Kaiser windowed sinc function not implemented for half-width of %d!'%(ireg,))
        raise

    freg = 2*ireg+1

    xOffset, zOffset = offset

    # Grid from 0 to freg-1
    Zi, Xi = np.mgrid[:freg,:freg] 

    # Distances from source point
    dZi = (zOffset + ireg - Zi)
    dXi = (xOffset + ireg - Xi)

    # Taper terms for decay function
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tZi = np.nan_to_num(np.sqrt(1 - (dZi / ireg)**2))
        tXi = np.nan_to_num(np.sqrt(1 - (dXi / ireg)**2))
        tZi[tZi == np.inf] = 0
        tXi[tXi == np.inf] = 0

    # Actual tapers for Kaiser window
    taperZ = bessi0(b*tZi) / bessi0(b)
    taperX = bessi0(b*tXi) / bessi0(b)

    # Windowed sinc responses in Z and X
    responseZ = np.sinc(dZi) * taperZ
    responseX = np.sinc(dXi) * taperX

    # Combined 2D source response
    result = responseX * responseZ

    return result

def srcVec(sLocs, terms, mesh, ireg, freeSurf):

    dx = mesh.hx[0]
    dz = mesh.hy[0]

    # Scale source based on the cellsize so that changing the grid doesn't
    # change the overall source amplitude
    srcScale = -mesh.hx[0]*mesh.hy[0]

    if getattr(terms, '__contains__', None) is None:
        terms = [terms]*len(sLocs)

    if ireg == 0:
        # Closest source point
        q = sp.lil_matrix((mesh.nN,1))
        q = q.ravel()

        for i in xrange(len(sLocs)):
            qI = SimPEG.Utils.closestPoints(mesh, sLocs[i], gridLoc='N')
            q[qI] += terms[i]/srcScale

    else:
        # Kaiser windowed sinc function

        freg = 2*ireg+1

        q = sp.lil_matrix((mesh.nNy+2*ireg, mesh.nNx+2*ireg), dtype=np.complex128)

        for i in xrange(len(sLocs)):
            qI = SimPEG.Utils.closestPoints(mesh, sLocs[i], gridLoc='N')[0]
            Zi, Xi = (qI / mesh.nNx, np.mod(qI, mesh.nNx))
            offset = (sLocs[i][0] - Xi * dx, sLocs[i][1] - Zi * dz)
            sourceRegion = KaiserWindowedSinc(ireg, offset)
            q[Zi:Zi+freg,Xi:Xi+freg] += terms[i] * sourceRegion / srcScale

        # Mirror and flip sign on terms that cross the free-surface boundary
        if freeSurf[0]:
            q[ireg:2*ireg,:]      = q[ireg:2:ireg,:].todense() - np.flipud(q[:ireg,:].todense())    # Top
        if freeSurf[1]:
            q[:,-2*ireg:-ireg]    = q[:,-2*ireg:-ireg].todense() - np.fliplr(q[:,-ireg:].todense())   # Right
        if freeSurf[2]:
            q[-2*ireg:-ireg,:]    = q[-2*ireg:-ireg,:].todense() - np.flipud(q[-ireg:,:].todense())   # Bottom
        if freeSurf[3]:
            q[:,ireg:2*ireg]      = q[:,ireg:2*ireg].todense() - np.fliplr(q[:,:ireg].todense())    # Left

        # Cut off edges
        q = sp.vstack((qi.T for qi in q[ireg:-ireg,ireg:-ireg]))

    return q

assumeConditions = lambda mesh: (getattr(mesh, 'ireg', DEFAULT_IREG), getattr(mesh, 'freeSurf', DEFAULT_FREESURF_BOUNDS))

class HelmRx(SimPEG.Survey.BaseRx):

    def __init__(self, locs, terms, **kwargs):

        self.terms = terms# if getattr(terms, '__contains__', None) is None else [terms]*locs.shape[0]

        SimPEG.Survey.BaseRx.__init__(self, locs.reshape((1,3)) if locs.ndim == 1 else locs, self.__class__.__name__, **kwargs)

    def getP(self, mesh, coeffs=None):

        if coeffs is not None:
            icoeffs = sp.diags(coeffs, 0)
        else:
            icoeffs = 1.

        if getattr(self, '_P', None) is None:
            if self.locs.shape[0] == 1:
                locterms = self.locs[0, ::2].reshape((1,2))
            else:
                locterms = self.locs[:,::2]

            ireg, freeSurf = assumeConditions(mesh)
            self._P = sp.vstack((srcVec(self.locs[i, ::2].reshape((1,2)), self.terms[i], mesh, ireg, freeSurf).T for i in xrange(locterms.shape[0])))

        return icoeffs * self._P

    # def getq(self, mesh, coeffs=None):

    #     if coeffs is not None:
    #         icoeffs = coeffs
    #     else:
    #         icoeffs = 1.

    #     if getattr(self, '_q', None) is None:
    #         if self.locs.shape[0] == 1:
    #             locterms = self.locs[0, ::2].reshape((1,2))
    #         else:
    #             locterms = self.locs[:, ::2]

    #         ireg, freeSurf = assumeConditions(mesh)
    #         self._q = srcVec(locterms, self.terms, mesh, ireg, freeSurf)

    #     q = self._q

    #     return q

    def getq(self, mesh, coeffs=None):

        return self.getP(mesh, coeffs).sum(axis=0).T

    def __getstate__(self):
        return {key: self.__dict__[key] for key in ['locs', 'terms', 'kwargs'] if key in self.__dict__}

    def __setstate__(self, d):
        if 'kwargs' in d:
            self.__init__(d['locs'], d['terms'], **d['kwargs'])
        else:
            self.__init__(d['locs'], d['terms'])

class HelmSrc(SimPEG.Survey.BaseSrc):

    rxPair = HelmRx

    def __init__(self, loc, term, rxList, **kwargs):

        # TODO: I would rather not store ireg and freeSurf in every single source and receiver!
        #       I feel like it actually makes more sense to have these as properties of the mesh.
        
        self.term = term

        SimPEG.Survey.BaseSrc.__init__(self, loc.reshape((1,3)), self.__class__.__name__, rxList, **kwargs)

    def getq(self, mesh):

        if getattr(self, '_q', None) is None:
            ireg, freeSurf = assumeConditions(mesh)
            self._q = srcVec(self.loc[0, ::2].reshape((1,2)), self.term, mesh, ireg, freeSurf)

        return self._q

    def _getcoeffs(self, ky):

        coeffs = []

        slocy = self.loc[0,1]
        for rx in self.rxList:
            rlocys = rx.locs[:, 1]
            dy = abs(slocy - rlocys)
            coeffs.append(np.cos(1*np.pi*ky*dy))

        return coeffs

    def getP(self, mesh, ky=None):

        if ky is None:
            inky = 0.
        else:
            inky = ky

        coeffslist = self._getcoeffs(inky)

        projs = (self.rxList[ir].getP(mesh, coeffslist[ir]).T for ir in xrange(len(self.rxList)))
        return sp.vstack(projs).T

    def getqback(self, mesh, terms, ky=None):

        if ky is None:
            inky = 0.
        else:
            inky = ky

        coeffslist = self._getcoeffs(inky)

        # qs = (self.rxList[ir].getq(mesh, terms*coeffslist[ir]).T for ir in xrange(len(self.rxList)))
        # return sp.vstack(qs).T
        return self.rxList[0].getq(mesh, terms*coeffslist[0])

    def __getitem__(self, sl):
        return self.rxList.__getitem__(sl)

    def __getstate__(self):
        return {key: self.__dict__[key] for key in ['loc', 'term', 'rxList', 'kwargs'] if key in self.__dict__}

    def __setstate__(self, d):
        if 'kwargs' in d:
            self.__init__(d['loc'], d['term'], d['rxList'], **d['kwargs'])
        else:
            self.__init__(d['loc'], d['term'], d['rxList'])

class SurveyHelm(SimPEG.Survey.BaseSurvey):

    srcPair = HelmSrc

    def __init__(self, dispatcher, **kwargs):

        self.dispatcher = dispatcher
        sc = self.dispatcher.systemConfig

        self.geom = sc.get('geom', None)

        if self.geom is not None:
            self.srcList = self.genSrc()
        else:
            self.srcList = None
        SimPEG.Survey.BaseSurvey.__init__(self, **kwargs)

    def genSrc(self, srcTerms=None, rxTerms=None):

        mode = self.geom['mode'].lower()

        if srcTerms is None:
            srcTerms = [1.]*self.geom['src'].shape[0]
        elif getattr(srcTerms, '__contains__', None) is None:
            srcTerms = [srcTerms]*self.geom['src'].shape[0]

        if rxTerms is None:
            rxTerms = [1.]*self.geom['rec'].shape[0]
        elif getattr(rxTerms, '__contains__', None) is None:
            rxTerms = [rxTerms]*self.geom['rec'].shape[0]

        if mode == 'relative':
            # Streamer relative to source location
            srcs = []
            for i, sloc in enumerate(self.geom['src']):
                rxs = [HelmRx(sloc + self.geom['rec'], rxTerms)]
                srcs.append(HelmSrc(sloc, srcTerms[i], rxs))

        elif mode == 'absolute':
            # Separate array in absolute coordinates for each source
            srcs = []
            for i, sloc in enumerate(self.geom['src']):
                rxs = [HelmRx(self.geom['rec'], rxTerms[i])]
                srcs.append(HelmSrc(sloc, srcTerms[i], rxs))

        else:
            # Fixed array common for all sources
            rxs = [HelmRx(self.geom['rec'], rxTerms)]
            srcs = [HelmSrc(sloc, srcTerms[i], rxs) for i, sloc in enumerate(self.geom['src'])]

        return srcs

    def projectFields(self, u=None):

        self.dispatcher.forward()

        return self.dispatcher.d




