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

assumeConditions = lambda mesh: (getattr(mesh, 'ireg', DEFAULT_IREG), getattr(mesh, 'freeSurf', DEFAULT_FREESURF_BOUNDS))

class HelmGeneral(object):

    def __init__(self, arrayTerms): #slocs, sterms, rlocs, rterms, **kwargs):

        mode = arrayTerms['mode'].lower() 

        if mode not in ['fixed', 'relative']:
            raise Exception('HelmGeneral objects only work with \'fixed\' or \'relative\' receiver arrays!')

        self.rel = mode is 'relative'

        slocs = arrayTerms['src']
        rlocs = arrayTerms['rec']
        sterms = arrayTerms.get('sterms', 1.)
        rterms = arrayTerms.get('rterms', 1.)

        self.nsrc = slocs.shape[0]
        self.nrec = rlocs.shape[0]

        if slocs.shape[1] == 3:
            self.sxz = np.array([slocs[:,0], slocs[:,2]]).T
            self.sy = slocs[:,1]
        elif slocs.shape[1] == 2:
            self.sxz = slocs
            self.sy = np.ones((self.nsrc,))
        else:
            raise Exception('Src location array must have either 2 or 3 columns')

        if rlocs.shape[1] == 3:
            self.rxz = np.array([rlocs[:,0], rlocs[:,2]]).T
            self.ry = rlocs[:,1]
        elif rlocs.shape[1] == 2:
            self.rxz = rlocs
            self.ry = np.ones((self.nrec,))
        else:
            raise Exception('Rec location array must have either 3 or 4 columns')

        if getattr(sterms, '__contains__', None) is not None:
            self.sterms = sterms
        else:
            self.sterms = sterms * np.ones((self.nsrc,))

        if getattr(rterms, '__contains__', None) is not None:
            self.rterms = rterms
        else:
            self.rterms = rterms * np.ones((self.nrec,))

    def getPGeneral(self, mesh, locs, terms):

        ireg, freeSurf = assumeConditions(mesh)

        dx = mesh.hx[0]
        dz = mesh.hy[0]
        nx = mesh.nNx
        nz = mesh.nNy

        # Scale source based on the cellsize so that changing the grid doesn't
        # change the overall source amplitude
        srcScale = -dx*dz

        if getattr(terms, '__contains__', None) is None:
            terms = np.array([terms]*locs.shape[0])

        qI = SimPEG.Utils.closestPoints(mesh, locs, gridLoc='N')

        if ireg == 0:
            # Closest gridpoint

            q = sp.coo_matrix((terms/srcScale, (np.arange(locs.shape[0]), qI)), shape=(locs.shape[0], mesh.nN))

        else:

            # Kaiser windowed sinc function

            freg = 2*ireg+1
            N = locs.shape[0]
            M = nz * nx
            nnz = N * freg**2
            lShift, sShift = np.mgrid[-ireg:ireg+1,-ireg:ireg+1]
            shift = lShift * nx + sShift

            entries = np.zeros((nnz,), dtype=np.complex128)
            columns =  np.zeros((nnz,))
            rows = np.zeros((nnz,))
            dptr = 0

            for i in xrange(N):
                Zi, Xi = (qI[i] / nx, np.mod(qI[i], nx))
                offset = (locs[i][0] - Xi * dx, locs[i][1] - Zi * dz)
                sourceRegion = KaiserWindowedSinc(ireg, offset)
                qshift = shift.copy()

                if Zi < ireg:
                    index = ireg-Zi
                    if freeSurf[2]:
                        lift = np.flipud(sourceRegion[:index,:])
                    
                    sourceRegion = sourceRegion[index:,:]
                    qshift = qshift[index:,:]

                    if freeSurf[2]:
                        sourceRegion[:index,:] -= lift

                if Zi > nz-ireg-1:
                    index = nz-ireg-1 - Zi
                    if freeSurf[0]: 
                        lift = np.flipud(sourceRegion[index:,:])

                    sourceRegion = sourceRegion[:index,:]
                    qshift = qshift[:index,:]

                    if freeSurf[0]:
                        sourceRegion[index:,:] -= lift

                if Xi < ireg:
                    index = ireg-Xi
                    if freeSurf[3]:
                        lift = np.fliplr(sourceRegion[:,:index])

                    sourceRegion = sourceRegion[:,index:]
                    qshift = qshift[:,index:]

                    if freeSurf[3]:
                        sourceRegion[:,:index] -= lift

                if Xi > nx-ireg-1:
                    index = nx-ireg-1 - Xi
                    if freeSurf[1]:
                        lift = np.fliplr(sourceRegion[:,index:])

                    sourceRegion = sourceRegion[:,:index]
                    qshift = qshift[:,:index]

                    if freeSurf[1]:
                        sourceRegion[:,index:] -= lift

                data = sourceRegion.ravel() * terms[i] / srcScale
                cols = qI[i] + qshift.ravel()
                dlen = data.shape[0]

                entries[dptr:dptr+dlen] = data
                columns[dptr:dptr+dlen] = cols
                rows[dptr:dptr+dlen] = i

                dptr += dlen

            q = sp.coo_matrix((entries[:dptr], (rows[:dptr],columns[:dptr])), shape=(N, M), dtype=np.complex128)

        return q

    def getRecP(self, mesh, src, coeffs=None, ky=0.):

        if coeffs is not None:
            icoeffs = sp.diags(coeffs, 0)
        else:
            icoeffs = 1.

        # TODO: Should this be 2*pi?
        if self.rel:
            dy = self.ry
        else:
            dy = self.ry - self.sy[src]

        shift = sp.diags(np.cos(np.pi * dy * ky), 0)
        icoeffs = shift * icoeffs

        if self.rel:
            if getattr(self, '_RecP', None) is None:
                self._RecP = {}
            if src not in self._RecP:
                self._RecP[src] = self.getPGeneral(mesh, self.rxz + self.sxz[src], self.rterms)
            recP = self._RecP[src]

        else:
            if getattr(self, '_RecP', None) is None:
                self._RecP = self.getPGeneral(mesh, self.rxz, self.rterms)
            recP = self._RecP

        return icoeffs * recP

    def getRecPAll(self, mesh, coeffs=None, ky=0., isrc=slice(None)):

        return (self.getRecP(mesh, isrc, coeffs[isrc] if coeffs is not None else None, ky) for isrc in range(self.nsrc)[isrc])

    def getSrcP(self, mesh, coeffs=None):

        if coeffs is not None:
            icoeffs = sp.diags(coeffs, 0)
        else:
            icoeffs = 1.

        if getattr(self, '_SrcP', None) is None:
            self._SrcP = self.getPGeneral(mesh, self.sxz, self.sterms)

        return (icoeffs * self._SrcP).tocsc()

    def __getstate__(self):
        return {key: self.__dict__[key] for key in ['nsrc', 'nrec', 'sxz', 'sy', 'rxz', 'ry', 'sterms', 'rterms', 'rel'] if key in self.__dict__}

    def __setstate__(self, d):
        for key in d:
            setattr(self, key, d[key])
