'''
Low-level programming constructs for Zephyr
'''
from __future__ import division

from galoshes import AttributeMapper
import numpy as np

class BaseModelDependent(AttributeMapper):
    '''
    AttributeMapper subclass that implements model-dependent properties,
    such as grid coordinates and free-surface conditions.
    '''

    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'nx':           (True,      None,           np.int64),
        'ny':           (False,     None,           np.int64),
        'nz':           (True,      None,           np.int64),
        'xorig':        (False,     '_xorig',       np.float64),
        'yorig':        (False,     '_xorig',       np.float64),
        'zorig':        (False,     '_zorig',       np.float64),
        'dx':           (False,     '_dx',          np.float64),
        'dy':           (False,     '_dx',          np.float64),
        'dz':           (False,     '_dz',          np.float64),
        'freeSurf':     (False,     '_freeSurf',    tuple),
    }

    @property
    def xorig(self):
        return getattr(self, '_xorig', 0.)

    @property
    def yorig(self):
        if hasattr(self, 'ny'):
            return getattr(self, '_yorig', 0.)
        else:
            raise AttributeError('%s object is not 3D'%(self.__class__.__name__,))

    @property
    def zorig(self):
        return getattr(self, '_zorig', 0.)

    @property
    def dx(self):
        return getattr(self, '_dx', 1.)

    @property
    def dy(self):
        if hasattr(self, 'ny'):
            return getattr(self, '_dy', self.dx)
        else:
            raise AttributeError('%s object is not 3D'%(self.__class__.__name__,))

    @property
    def dz(self):
        return getattr(self, '_dz', self.dx)

    @property
    def freeSurf(self):
        if getattr(self, '_freeSurf', None) is None:
            self._freeSurf = (False, False, False, False)
        return self._freeSurf

    @property
    def modelDims(self):
        if hasattr(self, 'ny'):
            return (self.nz, self.ny, self.nx)
        return (self.nz, self.nx)

    @property
    def nrow(self):
        return np.prod(self.modelDims)

    def toLinearIndex(self, vec):
        '''
        Gets the linear indices in the raveled model coordinates, given
        a <n by 2> array of n x,z coordinates or a <n by 3> array of
        n x,y,z coordinates.

        Args:
            vec (np.ndarray): Space coordinate array

        Returns:
            np.ndarray: Grid coordinate array
        '''

        if hasattr(self, 'ny'):
            return vec[:,0] * self.nx * self.ny + vec[:,1] * self.nx + vec[:,2]
        else:
            return vec[:,0] * self.nx + vec[:,1]

    def toVecIndex(self, lind):
        '''
        Gets the vectorized index for each linear index.

        Args:
            lind (np.ndarray): Grid coordinate array

        Returns:
            np.ndarray: nD grid coordinate array
        '''

        if hasattr(self, 'ny'):
            return np.array([lind // (self.nx * self.ny), np.mod(lind, self.nx), np.mod(lind, self.ny * self.nx)]).T
        else:
            return np.array([lind // self.nx, np.mod(lind, self.nx)]).T


class BaseAnisotropic(BaseModelDependent):

    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'theta':        (False,     '_theta',       np.float64),
        'eps':          (False,     '_eps',         np.float64),
        'delta':        (False,     '_delta',       np.float64),
    }

    @property
    def theta(self):
        if getattr(self, '_theta', None) is None:
            self._theta = np.zeros((self.nz, self.nx))

        if isinstance(self._theta, np.ndarray):
            return self._theta
        else:
            return self._theta * np.ones((self.nz, self.nx), dtype=np.float64)

    @property
    def eps(self):
        if getattr(self, '_eps', None) is None:
            self._eps = np.zeros((self.nz, self.nx))

        if isinstance(self._eps, np.ndarray):
            return self._eps
        else:
            return self._eps * np.ones((self.nz, self.nx), dtype=np.float64)

    @property
    def delta(self):
        if getattr(self, '_delta', None) is None:
            self._delta = np.zeros((self.nz, self.nx))

        if isinstance(self._delta, np.ndarray):
            return self._delta
        else:
            return self._delta * np.ones((self.nz, self.nx), dtype=np.float64)
