import numpy
import SimPEG

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
    dz = mesh.hz[0]

    # Scale source based on the cellsize so that changing the grid doesn't
    # change the overall source amplitude
    srcScale = -mesh.hx[0]*mesh.hz[0]

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

def srcTerm(sLocs, individual=True, terms=1):

    if individual and len(sLocs) > 1:
        result = []
        for i in xrange(len(sLocs)):
            result.append(srcVec([sLocs[i] if hasattr(sLocs, '__contains__') else sLocs], [terms[i]] if hasattr(terms, '__contains__') else [terms]))
    else:
        result = srcVec(sLocs if hasattr(sLocs, '__contains__') else [sLocs], terms if hasattr(terms, '__contains__') else [terms])

    return result 

class SurveyHelm(SimPEG.Survey.BaseSurvey):

    def __init__(self, txList, **kwargs):
        self.txList = txList
        self.ireg = ireg
        self.freeSurf = freeSurf
        SimPEG.Survey.BaseSurvey.__init__(self, **kwargs)

    def projectFields(self, u):
        data = []

        for i, tx in enumerate(self.txList):
            Proj = tx.rxList[0].getP(self.prob.mesh)    # Generate an operator to extract the data from the wavefield upon multiplication
            data.append(Proj*u[i])                      # Extract the data for that particular source and all receivers



# class Rec(object):

#     def __init__(self, parent, geometry, origin):

#         self._parent = parent
#         self._nsrc = parent.nsrc
#         self._mode = geometry['mode']
#         self._rec = geometry['rec']
#         self._origin = origin

#     def _getrec(self, i):
#         if self._mode == 'fixed':
#             return self._rec - self._origin
#         elif self._mode == 'relative':
#             return self._rec[i] - self._origin
#         else:
#             return None

#     def __getitem__(self, index):

#         if isinstance(index, slice):
#             return [self._getrec(i) for i in xrange(*index.indices(self._nsrc))]
#         else:
#             return self._getrec(index)

#     @property
#     def nrec(self):
#         if getattr(self, '_nrec', None) is None:
#             if self._mode == 'fixed':
#                 self._nrec = len(self._rec)
#             else:
#                 self._nrec = max((len(item) for item in self._rec))

#         return self._nrec

# class SeisLocator25D(object):

#     def __init__(self, geometry):

#         x0 = geometry.get('x0', 0.)
#         z0 = geometry.get('z0', 0.)
#         self._origin = numpy.array([x0, 0., z0]).reshape((1,3))

#         if len(geometry['src'].shape) < 2:
#             self.src = geometry['src'].reshape((1,3))
#         else:
#             self.src = geometry['src']
#         self.rec = Rec(self, geometry, self._origin)

#     def __call__(self, isrc, ky):

#         sloc = self.src[isrc,:].reshape((1,3)) - self._origin
#         rlocs = self.rec[isrc]
#         if len(rlocs.shape) < 2:
#             rlocs.shape = (1,3)
#         dy = abs(sloc[:,1] - rlocs[:,1])
#         coeffs = numpy.cos(2*numpy.pi*ky*dy)

#         return sloc[:,::2], rlocs[:,::2], coeffs

#     @property
#     def nsrc(self):
#         return len(self.src)

#     @property
#     def nrec(self):
#         return self.rec.nrec

class HelmTx(SimPEG.Survey.BaseTx):

    def __init__(self, loc, term, rxList, ireg, freeSurf, **kwargs):

        self.loc = loc
        self.term = term
        # TODO: I would rather not store ireg and freeSurf in every single source and receiver!
        self.ireg = ireg            
        self.freeSurf = freeSurf
        self.rxList = rxList
        self.kwargs = kwargs

        SimPEG.Survey.BaseTx.__init__(self)

    def getq(self, mesh):

        q = srcVec(self.loc, self.term, mesh, ireg, freeSurf)

class HelmRx(SimPEG.Survey.BaseRx):

    def __init__(self, locs, terms):

        self.locs = locs
        self.terms = terms

        SimPEG.Survey.BaseRx.__init__(self)

    @property
    def nD(self):
        ''' The number of receivers for this source '''
        return self.locs.shape[0]

    def getP(self, mesh):
        # P = mesh.getInterpolationMat(self.locs, 'CC')
        # ASSERT mesh == storedmesh
        P = SimPEG.Utils.sdiag()

        return P

