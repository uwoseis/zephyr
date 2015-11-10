
from .meta import BaseModelDependent
import numpy as np

class BaseSource(BaseModelDependent):
    
    pass


class FakeSource(BaseSource):
    
    def __call__(self, loc):
        
        return loc

    
class SimpleSource(BaseSource):
    
    def __init__(self, systemConfig):
        
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
        
        if hasattr(self, 'ny'):
            raise NotImplementedError('Sources not implemented for 3D case')
            dist = np.sqrt((self._x - loc[:,0])**2 + (self._y - loc[:,1])**2 + (self._z - loc[:,2])**2)
        else:
            dist = np.sqrt((self._x - loc[:,0])**2 + (self._z - loc[:,1])**2)
            
        return dist
    
    def __call__(self, loc):
        
        dist = self.dist(loc)
        srcterm = 1.*(dist == dist.min())
        q = srcterm.ravel() / srcterm.sum()
        
        return q
    
    
class StackedSimpleSource(SimpleSource):

    def __call__(self, loc):

        q = super(StackedSimpleSource, self).__call__(loc)
        return np.hstack([q, np.zeros(self._x.size, dtype=np.complex128)])


class KaiserSource(SimpleSource):
    
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
        '''

        from scipy.special import i0 as bessi0
        import warnings

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
    
    def __call__(self, sLocs, terms=None):
        
        if terms is None:
            terms = np.ones((len(sLocs),), dtype=np.complex128)

        q = np.zeros((self.nz, self.nx), dtype=np.complex128)

        # Scale source based on the cellsize so that changing the grid doesn't
        # change the overall source amplitude
        srcScale = self.dx*self.dz
        
        ireg = self.ireg

        if ireg == 0:
            # Closest source point
            q = q.ravel()

            for i in xrange(len(sLocs)):
                qI = self.toLinearIndex(self.dist(sLocs))
                q[qI] += terms[i]/srcScale

        else:
            # Kaiser windowed sinc function

            freg = 2*ireg+1
            q = np.pad(q, ireg, mode='constant')

            for i in xrange(len(sLocs)):
                Zi, Xi = self.toVecIndex(np.argmin(self.dist(sLocs[i].reshape((1,2)))))
                offset = (sLocs[i][0] - Xi * self.dx, sLocs[i][1] - Zi * self.dz)
                sourceRegion = self.kws(offset)
                q[Zi:Zi+freg,Xi:Xi+freg] += terms[i] * sourceRegion / srcScale

            # Mirror and flip sign on terms that cross the free-surface boundary
            if self.freeSurf[0]:
                q[ireg:2*ireg,:]      -= np.flipud(q[:ireg,:])    # Top
            if self.freeSurf[1]:
                q[:,-2*ireg:-ireg]    -= np.fliplr(q[:,-ireg:])   # Right
            if self.freeSurf[2]:
                q[-2*ireg:-ireg,:]    -= np.flipud(q[-ireg:,:])   # Bottom
            if self.freeSurf[3]:
                q[:,ireg:2*ireg]      -= np.fliplr(q[:,:ireg])    # Left

            # Cut off edges
            q = q[ireg:-ireg,ireg:-ireg].ravel()

        return q
    
    @property
    def ireg(self):
        return getattr(self, '_ireg', 4)
