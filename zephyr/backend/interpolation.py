
import warnings
from scipy.special import i0 as bessi0
import numpy as np
from .meta import BaseModelDependent, BaseSCCache

from scipy.interpolate import RectBivariateSpline

class BaseGridInterpolator(BaseModelDependent, BaseSCCache):
    '''
    Base class for interpolation between two regular grids.
    Defines helper functions and properties to produce regular
    grids as arrays. Also can create its own transpose.
    '''

    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'scale':        (True,      None,           np.float64),
        'eCons':        (False,     '_eCons',       bool),
    }

    @property
    def eCons(self):
        return getattr(self, '_eCons', False)

    @staticmethod
    def genGrid(nx, nz, dx, dz, xorig, zorig):

        Zi, Xi = np.mgrid[0:nz, 0:nx]
        Zi = Zi * dz + zorig
        Xi = Xi * dx + xorig

        return Zi, Xi

    @property
    def nativeGrid(self):
        if not hasattr(self, '_grid'):
            Zi, Xi = self.genGrid(
                self.nx,
                self.nz,
                self.dx,
                self.dz,
                self.xorig,
                self.zorig
            )
            self._nativeGrid = Zi, Xi
        return self._nativeGrid

    @property
    def Xg(self):
        return self.nativeGrid[1]

    @property
    def Zg(self):
        return self.nativeGrid[0]

    @property
    def Z(self):
        return np.linspace(
            self.zorig,
            self.zorig + self.dz * (self.nz-1),
            self.nz)

    @property
    def X(self):
        return np.linspace(
            self.xorig,
            self.xorig + self.dx * (self.nx-1),
            self.nx)

    @property
    def snx(self):
        return np.round(self.nx / self.scale)

    @property
    def snz(self):
        return np.round(self.nz / self.scale)

    @property
    def sdx(self):
        return self.dx * self.scale

    @property
    def sdz(self):
        return self.dz * self.scale

    @property
    def scaledGrid(self):
        if not hasattr(self, '_grid'):
            Zi, Xi = self.genGrid(
                self.snx,
                self.snz,
                self.sdx,
                self.sdz,
                self.xorig,
                self.zorig)
            self._scaledGrid = Zi, Xi
        return self._scaledGrid

    @property
    def sXg(self):
        return self.scaledGrid[1]

    @property
    def sZg(self):
        return self.scaledGrid[0]

    @property
    def sZ(self):
        return np.linspace(
            self.zorig,
            self.zorig + self.sdz * (self.snz-1),
            self.snz)

    @property
    def sX(self):
        return np.linspace(
            self.xorig,
            self.xorig + self.sdx * (self.snx-1),
            self.snx)

    @property
    def compression(self):
        return self.scale**2

    @property
    def shape(self):
        return (self.snx * self.snz, self.nx * self.nz)

    @property
    def T(self):
        if not hasattr(self, '_T'):
            systemConfigT = {key: self.systemConfig[key] for key in self.systemConfig}
            systemConfigT['scale'] = 1. / self.scale
            systemConfigT['nx'] = self.snx
            systemConfigT['nz'] = self.snz
            systemConfigT['dx'] = self.sdx
            systemConfigT['dz'] = self.sdz
            self._T = self.__class__(systemConfigT)

            # assert self._T.shape[0] == self.shape[1]
            # assert self._T.shape[1] == self.shape[0]

        return self._T

    @property
    def scaleUpdate(self):

        update = {
            'nx':   self.snx,
            'nz':   self.snz,
            'dx':   self.sdx,
            'dz':   self.sdz,
        }

        return update

    def __mul__(self, value):

        raise NotImplementedError

    def __call__(self, value):

        return self * value


class SplineGridInterpolator(BaseGridInterpolator):
    '''
    Interpolator class that uses bivariate splines for interpolation.
    '''

    def __mul__(self, rhs):

        if self.shape[0] == self.shape[1]:
            return rhs

        if rhs.ndim == 2:
            output = np.zeros((self.shape[0], rhs.shape[1]), dtype=rhs.dtype.type)
            for i in xrange(rhs.shape[1]):
                output[:,i] = self * rhs[:,i]
            return output

        elif rhs.ndim > 2:
            raise NotImplementedError('%s does not support %dD inputs'%(self.__class__.__name__, rhs.ndim))

        if issubclass(rhs.dtype.type, np.complex):
            return (self * rhs.real) + 1j * (self * rhs.imag)

        rbs = RectBivariateSpline(self.Z, self.X, rhs.reshape((self.nz, self.nx)))
        result = rbs(self.sZ, self.sX, grid=True)
        if self.eCons:
            result = result * self.compression
        return result.ravel()

