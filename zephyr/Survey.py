import numpy
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

    KaiserWindowedSince(ireg, offset) --> 2D ndarray of size (2*ireg+1, 2*ireg+1)
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
    Zi, Xi = numpy.mgrid[:freg,:freg] 

    # Distances from source point
    dZi = (zOffset + ireg - Zi)
    dXi = (xOffset + ireg - Xi)

    # Taper terms for decay function
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tZi = numpy.nan_to_num(numpy.sqrt(1 - (dZi / ireg)**2))
        tXi = numpy.nan_to_num(numpy.sqrt(1 - (dXi / ireg)**2))
        tZi[tZi == numpy.inf] = 0
        tXi[tXi == numpy.inf] = 0

    # Actual tapers for Kaiser window
    taperZ = bessi0(b*tZi) / bessi0(b)
    taperX = bessi0(b*tXi) / bessi0(b)

    # Windowed sinc responses in Z and X
    responseZ = numpy.sinc(dZi) * taperZ
    responseX = numpy.sinc(dXi) * taperX

    # Combined 2D source response
    result = responseX * responseZ

    return result

def srcVec(sLocs, terms, mesh, ireg, freeSurf):

    q = numpy.zeros((mesh.nNy, mesh.nNx), dtype=numpy.complex128)
    dx = mesh.hx[0]
    dz = mesh.hy[0]

    # Scale source based on the cellsize so that changing the grid doesn't
    # change the overall source amplitude
    srcScale = -mesh.hx[0]*mesh.hy[0]

    if ireg == 0:
        # Closest source point
        q = q.ravel()

        for i in xrange(len(sLocs)):
            qI = SimPEG.Utils.closestPoints(mesh, sLocs[i], gridLoc='N')
            q[qI] += terms[i]/srcScale

    else:
        # Kaiser windowed sinc function

        freg = 2*ireg+1
        q = numpy.pad(q, ireg, mode='constant')

        for i in xrange(len(sLocs)):
            qI = SimPEG.Utils.closestPoints(mesh, sLocs[i], gridLoc='N')
            Zi, Xi = (qI / mesh.nNx, numpy.mod(qI, mesh.nNx))
            offset = (sLocs[i][0] - Xi * dx, sLocs[i][1] - Zi * dz)
            sourceRegion = KaiserWindowedSinc(ireg, offset)
            q[Zi:Zi+freg,Xi:Xi+freg] += terms[i] * sourceRegion / srcScale

        # Mirror and flip sign on terms that cross the free-surface boundary
        if freeSurf[0]:
            q[ireg:2*ireg,:]      -= numpy.flipud(q[:ireg,:])    # Top
        if freeSurf[1]:
            q[:,-2*ireg:-ireg]    -= numpy.fliplr(q[:,-ireg:])   # Right
        if freeSurf[2]:
            q[-2*ireg:-ireg,:]    -= numpy.flipud(q[-ireg:,:])   # Bottom
        if freeSurf[3]:
            q[:,ireg:2*ireg]      -= numpy.fliplr(q[:,:ireg])    # Left

        # Cut off edges
        q = q[ireg:-ireg,ireg:-ireg].ravel()

    return q

srcVecs = lambda sLocs, terms, mesh, ireg, freeSurf: [srcVec([sLocs[i] if hasattr(sLocs, '__contains__') else sLocs], [terms[i]] if hasattr(terms, '__contains__') else [terms], mesh, ireg, freeSurf) for i in xrange(len(sLocs))]

class HelmRx(SimPEG.Survey.BaseRx):

    def __init__(self, locs, terms, ireg, freeSurf, **kwargs):

        self.terms = terms
        self.ireg = ireg
        self.freeSurf = freeSurf

        SimPEG.Survey.BaseRx.__init__(self, locs.reshape((1,3)) if locs.ndim == 1 else locs, self.__class__.__name__, **kwargs)

    def getP(self, mesh, coeffs=None):

        return [q.T for q in self.getq(mesh, coeffs)]

    def getq(self, mesh, coeffs=None):

        if coeffs is not None:
            icoeffs = coeffs
        else:
            icoeffs = 1.

        if self.locs.shape[0] == 1:
            locterms = self.locs[0, ::2].reshape((1,2))
        else:
            locterms = self.locs[:, ::2]

        q = srcVecs(locterms, icoeffs*self.terms, mesh, self.ireg, self.freeSurf)

        return q

    def __getstate__(self):
        return {key: self.__dict__[key] for key in ['locs', 'terms', 'ireg', 'freeSurf', 'kwargs'] if key in self.__dict__}

    def __setstate__(self, d):
        if 'kwargs' in d:
            self.__init__(d['locs'], d['terms'], d['ireg'], d['freeSurf'], **d['kwargs'])
        else:
            self.__init__(d['locs'], d['terms'], d['ireg'], d['freeSurf'])

class HelmTx(SimPEG.Survey.BaseTx):

    rxPair = HelmRx

    def __init__(self, loc, term, rxList, ireg, freeSurf, **kwargs):

        # TODO: I would rather not store ireg and freeSurf in every single source and receiver!
        #       I feel like it actually makes more sense to have these as properties of the mesh.
        
        self.term = term
        self.ireg = ireg            
        self.freeSurf = freeSurf

        SimPEG.Survey.BaseTx.__init__(self, loc.reshape((1,3)), self.__class__.__name__, rxList, **kwargs)

    def getq(self, mesh):

        q = srcVecs(self.loc[0, ::2].reshape((1,2)), self.term, mesh, self.ireg, self.freeSurf)[0]

        return q

    def _getcoeffs(self, ky):

        coeffs = []

        slocy = self.loc[0,1]
        for rx in self.rxList:
            rlocys = rx.locs[:, 1]
            dy = abs(slocy - rlocys)
            coeffs.append(numpy.cos(1*numpy.pi*ky*dy))

        return coeffs

    def getP(self, mesh, ky=None):

        if ky is None:
            inky = 0.
        else:
            inky = ky

        result = []
        coeffslist = self._getcoeffs(inky)

        for ir in xrange(len(self.rxList)):
            rx = self.rxList[ir]
            coeffs = coeffslist[ir]
            result.append(rx.getP(mesh, coeffs))

        return result

    def getqback(self, mesh, terms, ky=None):

        if ky is None:
            inky = 0.
        else:
            inky = ky

        coeffs = self._getcoeffs(inky)

        # Loop over all receivers in self.rxList
        # Get the results, possibly for more than one component, times the appropriate term
        # Add all of these together to produce a single RHS vector for backpropagation
        q = reduce(numpy.add, (reduce(numpy.add, rx.getq(mesh, terms[i]*coeffs[i])) for i, rx in enumerate(self.rxList)))

        return q

    def __getitem__(self, sl):
        return self.rxList.__getitem__(sl)

    def __getstate__(self):
        return {key: self.__dict__[key] for key in ['loc', 'term', 'rxList', 'ireg', 'freeSurf', 'kwargs'] if key in self.__dict__}

    def __setstate__(self, d):
        if 'kwargs' in d:
            self.__init__(d['loc'], d['term'], d['rxList'], d['ireg'], d['freeSurf'], **d['kwargs'])
        else:
            self.__init__(d['loc'], d['term'], d['rxList'], d['ireg'], d['freeSurf'])

class SurveyHelm(SimPEG.Survey.BaseSurvey):

    txPair = HelmTx

    def __init__(self, dispatcher, **kwargs):

        sc = dispatcher.systemConfig

        self.ireg = sc.get('ireg', DEFAULT_IREG)
        self.freeSurf = sc.get('freeSurf', DEFAULT_FREESURF_BOUNDS)

        self.geom = sc.get('geom', None)

        if self.geom is not None:
            self.txList = self.genTx()
        else:
            self.txList = None
        SimPEG.Survey.BaseSurvey.__init__(self, **kwargs)

    def genTx(self, txTerms=None, rxTerms=None):

        mode = self.geom['mode'].lower()

        if mode == 'relative':
            # Streamer relative to source location
            txs = []
            for i, sloc in enumerate(self.geom['src']):
                rxs = [HelmRx(sloc + rloc, rxTerms[j] if rxTerms is not None else 1., self.ireg, self.freeSurf) for j, rloc in enumerate(self.geom['rec'])]
                txs.append(HelmTx(sloc, txTerms[i] if txTerms is not None else 1., rxs, self.ireg, self.freeSurf))

        elif mode == 'absolute':
            # Separate array in absolute coordinates for each source
            txs = []
            for i, sloc in enumerate(self.geom['src']):
                rxs = [HelmRx(rloc, rxTerms[i][j] if rxTerms is not None else 1., self.ireg, self.freeSurf) for j, rloc in enumerate(self.geom['rec'][i])]
                txs.append(HelmTx(sloc, txTerms[i] if txTerms is not None else 1., rxs, self.ireg, self.freeSurf))

        else:
            # Fixed array common for all sources
            rxs = [HelmRx(loc, rxTerms[j] if rxTerms is not None else 1., self.ireg, self.freeSurf) for j, loc in enumerate(self.geom['rec'])]
            txs = [HelmTx(loc, txTerms[i] if txTerms is not None else 1., rxs, self.ireg, self.freeSurf) for i, loc in enumerate(self.geom['src'])]

        return txs

    def projectFields(self, u):

        data = []
        for i, tx in enumerate(self.txList):
            Proj = tx.rxList[0].getP(self.prob.mesh)    # Generate an operator to extract the data from the wavefield upon multiplication
            data.append(Proj*u[i])                      # Extract the data for that particular source and all receivers

        data = DataObject(self.prob, self.txList)




