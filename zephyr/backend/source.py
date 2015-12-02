'''
Source-generating routines for Zephyr
'''

from .meta import BaseModelDependent
import warnings
import numpy as np
import scipy.sparse as sp
from scipy.special import i0 as bessi0


class BaseSource(BaseModelDependent):
    'Trivial base class for sources'
    
    pass


class FakeSource(BaseSource):
    'Source that does nothing (for use with analytical systems)'
    
    def __call__(self, loc):
        
        return loc

    
class SimpleSource(BaseSource):
    '''
    A basic source-implementing class. This takes configuration
    from a systemConfig object, and calling it will return a set
    of right-hand-side vectors for the appropriate space locations.
    '''
    
    def __init__(self, systemConfig):
        'Initialize based on systemConfig'
        
        super(BaseSource, self).__init__(systemConfig)
        
        if hasattr(self, 'ny'):
            raise NotImplementedError('Sources not implemented for 3D case')
            self._z, self._y, self._x = np.mgrid[
                self.zorig : self.dz * self.nz : self.dz,
                self.yorig : self.dy * self.ny : self.dy,
                self.xorig : self.dx * self.nx : self.dx
            ]
        else:
            self._z, self._x = np.mgrid[
                self.zorig : self.dz * self.nz : self.dz,
                self.xorig : self.dx * self.nx : self.dx
            ]
    
    def dist(self, loc):
        '''
        Calculates the distance of each gridpoint from the source locations.
        
        Args:
            loc (np.ndarray): Source locations in space
        
        Returns:
            np.ndarray: The distance from each gridpoint
        '''
        
        nsrc = len(loc)
        if hasattr(self, 'ny'):
            raise NotImplementedError('Sources not implemented for 3D case')
            dist = np.sqrt((self._x.reshape((1, self.nz, self.ny, self.nx)) - loc[:,0].reshape((nsrc, 1, 1, 1)))**2
                         + (self._y.reshape((1, self.nz, self.ny, self.nx)) - loc[:,1].reshape((nsrc, 1, 1, 1)))**2
                         + (self._z.reshape((1, self.nz, self.ny, self.nx)) - loc[:,2].reshape((nsrc, 1, 1, 1)))**2)
        else:
            dist = np.sqrt((self._x.reshape((1, self.nz, self.nx)) - loc[:,0].reshape((nsrc, 1, 1)))**2
                         + (self._z.reshape((1, self.nz, self.nx)) - loc[:,1].reshape((nsrc, 1, 1)))**2)
            
        return dist
    
    def vecIndexOf(self, loc):
        'The vector index of each source location'
        return self.toVecIndex(self.linIndexOf(loc))
    
    def linIndexOf(self, loc):
        'The linear index of each source location'
        nsrc = loc.shape[0]
        
        dists = self.dist(loc).reshape((nsrc, self.nrow))
        return np.argmin(dists, axis=1)
    
    def __call__(self, loc):
        '''
        Given source locations, return a set of right-hand-side vectors.
        
        Args:
            loc (np.ndarray): Source locations in space
        
        Returns:
            np.ndarray: Dense right-hand side vectors
        '''
        
        nsrc = loc.shape[0]
        q = np.zeros((nsrc, self.nrow), dtype=np.complex128)
        
        for i, index in enumerate(self.linIndexOf(loc)):
            q[i,index] = 1.
        
        return q.T
    
    
class StackedSimpleSource(SimpleSource):
    '''
    A SimpleSource subclass that returns source vectors twice the size,
    augmented with zeros.
    '''

    def __call__(self, loc):

        q = super(StackedSimpleSource, self).__call__(loc)
        return np.vstack([q, np.zeros(q.shape, dtype=np.complex128)])


class SparseKaiserSource(SimpleSource):
    '''
    A source-implementing class suitable for use in production codes.
    SparseKaiser source takes a systemConfig dictionary. When called,
    it returns a SciPy sparse matrix of source vectors, suitable for
    efficient use and/or caching. The source vectors make use of
    Graham Hicks's Kaiser-Windowed Sinc Function sources, to allow
    for interpolating between grid points.
    '''
    
    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'ireg':         (False,     '_ireg',        np.int64),
        'freeSurf':     (False,     '_freeSurf',    tuple),
    }
    
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

    def kws(self, offset):
        '''
        Finds 2D source terms to approximate a band-limited point source, based on
        Hicks, Graham J. (2002) Arbitrary source and receiver positioning in finite-difference
            schemes using Kaiser windowed sinc functions. Geophysics (67) 1, 156-166.
        KaiserWindowedSinc(ireg, offset) --> 2D ndarray of size (2*ireg+1, 2*ireg+1)
        Input offset is the 2D offsets in fractional gridpoints between the source location and
        the nearest node on the modelling grid.
        
        Args:
            offset (tuple): Distance of the centre of the source region from the true source
                            point (in 2D coordinates, in units of cells).
        
        Returns:
            np.ndarray: Interpolated source region of size (2*ireg+1, 2*ireg+1)
        '''

        try:
            b = self.HC_KAISER.get(self.ireg)
        except KeyError:
            print('Kaiser windowed sinc function not implemented for half-width of %d!'%(ireg,))
            raise
        
        freg = 2*self.ireg+1

        xOffset, zOffset = offset

        # Grid from 0 to freg-1
        Zi, Xi = np.mgrid[:freg,:freg] 

        # Distances from source point
        dZi = (zOffset + self.ireg - Zi)
        dXi = (xOffset + self.ireg - Xi)

        # Taper terms for decay function
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tZi = np.nan_to_num(np.sqrt(1 - (dZi / self.ireg)**2))
            tXi = np.nan_to_num(np.sqrt(1 - (dXi / self.ireg)**2))
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
    
    def __call__(self, sLocs):
        '''
        Given source locations, return a set of right-hand-side vectors.
        
        Args:
            sLocs (np.ndarray): Source locations in space
        
        Returns:
            np.ndarray: Dense right-hand side vectors
        '''
        
        ireg = self.ireg
        freeSurf = self.freeSurf
        N = sLocs.shape[0]
        M = self.nz * self.nx
        
        # Scale source based on the cellsize so that changing the grid doesn't
        # change the overall source amplitude   
        srcScale = 1. / (self.dx * self.dz)

        qI = self.linIndexOf(sLocs)

        if ireg == 0:
            # Closest gridpoint

            q = sp.coo_matrix((srcScale, (np.arange(N), qI)), shape=(N, M))

        else:

            # Kaiser windowed sinc function

            freg = 2*ireg+1

            nnz = N * freg**2
            lShift, sShift = np.mgrid[-ireg:ireg+1,-ireg:ireg+1]
            shift = lShift * self.nx + sShift

            entries = np.zeros((nnz,), dtype=np.complex128)
            columns =  np.zeros((nnz,))
            rows = np.zeros((nnz,))
            dptr = 0

            for i in xrange(N):
                Zi, Xi = (qI[i] / self.nx, np.mod(qI[i], self.nx))
                offset = (sLocs[i][0] - Xi * self.dx, sLocs[i][1] - Zi * self.dz)
                sourceRegion = self.kws(offset)
                qshift = shift.copy()

                if Zi < ireg:
                    index = ireg-Zi
                    if freeSurf[2]:
                        lift = np.flipud(sourceRegion[:index,:])
                    
                    sourceRegion = sourceRegion[index:,:]
                    qshift = qshift[index:,:]

                    if freeSurf[2]:
                        sourceRegion[:index,:] -= lift

                if Zi > self.nz-ireg-1:
                    index = self.nz-ireg-1 - Zi
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

                if Xi > self.nx-ireg-1:
                    index = self.nx-ireg-1 - Xi
                    if freeSurf[1]:
                        lift = np.fliplr(sourceRegion[:,index:])

                    sourceRegion = sourceRegion[:,:index]
                    qshift = qshift[:,:index]

                    if freeSurf[1]:
                        sourceRegion[:,index:] -= lift

                data = srcScale * sourceRegion.ravel()
                cols = qI[i] + qshift.ravel()
                dlen = data.shape[0]

                entries[dptr:dptr+dlen] = data
                columns[dptr:dptr+dlen] = cols
                rows[dptr:dptr+dlen] = i

                dptr += dlen

            q = sp.coo_matrix((entries[:dptr], (rows[:dptr],columns[:dptr])), shape=(N, M), dtype=np.complex128)

        return q.T
    
    @property
    def ireg(self):
        'Half-width of the source region'
        return getattr(self, '_ireg', 4)
    
class KaiserSource(SparseKaiserSource):
    '''
    A simple wrapper class that generates dense sources
    using SparseKaiserSource.
    '''
    
    def __call__(self, sLocs):
        
        q = super(KaiserSource, self).__call__(sLocs)
        return q.toarray()


