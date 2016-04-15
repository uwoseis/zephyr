'''
Distribution wrappers for composite problems
'''
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import super
from future import standard_library
standard_library.install_aliases()
from builtins import zip
from builtins import range

from galoshes import SCFilter, BaseSCCache
import numpy as np
from .discretization import DiscretizationWrapper
from .interpolation import SplineGridInterpolator
from .base import BaseModelDependent

try:
    import multiprocessing
except ImportError:
    PARALLEL = False
else:
    PARALLEL = True

PARTASK_TIMEOUT = None

class BaseDist(DiscretizationWrapper):

    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'Disc':         (True,      '_Disc',        None),
        'parallel':     (False,     '_parallel',    bool),
        'nWorkers':     (False,     '_nWorkers',    np.int64),
        'remDists':     (False,     None,           list),
    }

    maskKeys = {'remDists'}

    @property
    def remDists(self):
        'Remaining distributor objects in the call graph'

        return getattr(self, '_remDists', [])
    @remDists.setter
    def remDists(self, value):
        if value:
            self._DiscOverride = value.pop(0)
        self._remDists = value

    @property
    def Disc(self):
        'The discretization to instantiate'

        return getattr(self, '_DiscOverride', self._Disc)

    @property
    def addFields(self):
        'Returns additional fields for the subProblem systemConfigs'

        return {'remDists': self.remDists}

    @property
    def systemConfig(self):
        self._systemConfig.update(self.remDists)
        return self._systemConfig
    @systemConfig.setter
    def systemConfig(self, value):
        self._systemConfig = value


class BaseMPDist(BaseDist):

    maskKeys = {'parallel'}

    @property
    def parallel(self):
        'Determines whether to operate in parallel'

        return PARALLEL and getattr(self, '_parallel', True)

    @property
    def pool(self):
        'Returns a configured multiprocessing Pool'

        if self.parallel:
            if not hasattr(self, '_pool'):
                self._pool = multiprocessing.Pool(self.nWorkers)
            return self._pool

        else:
            raise Exception('Cannot start parallel pool; multiprocessing seems to be unavailable')

    @property
    def nWorkers(self):
        'Returns the configured number of parallel workers'

        return min(getattr(self, '_nWorkers', 100), self.cpuCount)

    @property
    def cpuCount(self):
        'Returns the multiprocessing CPU count'

        if self.parallel:
            return multiprocessing.cpu_count()
        else:
            return 1

    @property
    def addFields(self):
        'Returns additional fields for the subProblem systemConfigs'

        fields = super(BaseMPDist, self).addFields

        remCap = self.cpuCount // self.nWorkers
        if (self.nWorkers < self.cpuCount) and remCap > 1:

            fields.update({'parallel': True, 'nWorkers': remCap})

        return fields

    def __mul__(self, rhs):
        '''
        Carries out the multiplication of the composite system
        by the right-hand-side vector(s).

        Args:
            rhs (array-like or list thereof): Source vectors

        Returns:
            u (iterator over np.ndarrays): Wavefields
        '''

        if isinstance(rhs, list):
            getRHS = lambda i: rhs[i]
        else:
            getRHS = lambda i: rhs

        if self.parallel:
            plist = []
            for i, sub in enumerate(self.subProblems):

                p = self.pool.apply_async(sub, (getRHS(i),))
                plist.append(p)

            u = (self.scaleTerm*p.get(PARTASK_TIMEOUT) for p in plist)

        else:
            u = (self.scaleTerm*(sub*getRHS(i)) for i, sub in enumerate(self.subProblems))

        return u

    @property
    def factors(self):
        # What this does:
        #   Return True if there is a pool defined
        #   If there isn't, check to see if _subProblems exists; if it doesn't, return False
        #   If _subProblems *does* exist, check each subProblem to see if it has matrix factors.
        #   If any subProblem has factors, return True.
        return hasattr(self, '_pool') or not ((not hasattr(self, '_subProblems')) or (not any((sp.factors for sp in self.subProblems))))
    @factors.deleter
    def factors(self):
        if hasattr(self, '_pool'):
            self._pool.close()
            del self._pool
        if hasattr(self, '_subProblems'):
            for sp in self.subProblems:
                del sp.factors

    def __del__(self):
        del self.factors


class BaseIPYDist(BaseDist):

    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'profile':      (False,     '_profile',     str),
    }

    maskKeys = {'profile'}

    @property
    def profile(self):
        'Returns the IPython parallel profile'

        return getattr(self, '_profile', 'default')

    @property
    def pClient(self):
        'Returns the IPython parallel client'

        if not hasattr(self, '_pClient'):
            from ipyparallel import Client
            self._pClient = Client(self.profile)
        return self._pClient

    @property
    def dView(self):
        'Returns a direct (multiplexing) view on the IPython parallel client'

        if not hasattr(self, '_dView'):
            self._dView = self.pClient[:]
        return self._dView

    @property
    def lView(self):
        'Returns a load-balanced view on the IPython parallel client'

        if not hasattr(self, '_lView'):
            self._lView = self.pClient.load_balanced_view()
        return self._lView

    @property
    def nWorkers(self):
        'Returns the configured number of parallel workers'

        return len(self.pClient.ids)


class MultiFreq(BaseMPDist):
    '''
    Wrapper to carry out forward-modelling using the stored
    discretization over a series of frequencies.
    '''

    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'freqs':        (True,      None,           list),
    }

    maskKeys = {'freqs'}

    @property
    def spUpdates(self):
        'Updates for frequency subProblems'

        vals = []
        for freq in self.freqs:
            spUpdate = {'freq': freq}
            spUpdate.update(self.addFields)
            vals.append(spUpdate)
        return vals


class ViscoMultiFreq(MultiFreq, BaseModelDependent):
    '''
    Wrapper to carry out forward-modelling using the stored
    discretization over a series of frequencies. Preserves
    causality by modelling velocity dispersion in the
    presence of a non-infinite Q model.
    '''

    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'c':            (True,      None,           np.float64),
        'Q':            (False,     None,           np.float64),
        'freqBase':     (False,     None,           np.float64),
    }

    maskKeys = {'freqs', 'c', 'Q', 'freqBase'}

    @staticmethod
    def _any(criteria):
        'Check for criteria on a scalar or vector'

        if type(criteria) in (bool, np.bool_):
            return criteria
        else:
            return np.any(criteria)

    @property
    def freqBase(self):
        return getattr(self, '_freqBase', 0.)
    @freqBase.setter
    def freqBase(self, value):
        assert value >= 0
        self._freqBase = value

    @property
    def Q(self):

        # NB: QC says to merge these two statements. Do not do that. The code
        #     "hasattr(self, '_Q') and not isinstance(self._Q, np.ndarray)"
        #     does not behave the same way in terms of when the 'else' statement
        #     is fired.

        if hasattr(self, '_Q'):
            if not isinstance(self._Q, np.ndarray):
                return self._Q * np.ones((self.nz, self.nx), dtype=np.float64)
        else:
            self._Q = np.inf

        return self._Q
    @Q.setter
    def Q(self, value):
        criteria = value <= 0
        try:
            assert not criteria
        except TypeError:
            assert not self._any(criteria)
        self._Q = value

    @property
    def disperseFreqs(self):
        return self._any(self.Q != np.inf) and (self.freqBase > 0)

    @property
    def spUpdates(self):
        'Updates for frequency subProblems'

        vals = []
        if self.disperseFreqs:
            for freq in self.freqs:
                fact = 1. + (np.log(freq / self.freqBase) / (np.pi * self.Q))
                assert not self._any(fact < 0.1)
                cR = fact * self.c
                c = cR + (0.5j * cR / self.Q) # NB: + b/c of FT convention

                spUpdate = {
                    'freq': freq,
                    'c':    c,
                }
                spUpdate.update(self.addFields)
                vals.append(spUpdate)

        else:
            for freq in self.freqs:
                c = self.c.ravel() + (0.5j * self.c.ravel() / self.Q.ravel()) # NB: + b/c of FT convention
                spUpdate = {
                    'freq': freq,
                    'c':    c,
                }
                spUpdate.update(self.addFields)
                vals.append(spUpdate)

        return vals


class SerialMultiFreq(MultiFreq):
    '''
    Wrapper to carry out forward-modelling using the stored
    discretization over a series of frequencies. Enforces
    serial execution.
    '''

    @property
    @staticmethod
    def parallel():
        'Determines whether to operate in parallel'

        return False

    @property
    @staticmethod
    def addFields():
        'Returns additional fields for the subProblem systemConfigs'

        return {}


class MultiGridMultiFreq(MultiFreq, BaseModelDependent):
    '''
    Wrapper to carry out forward-modelling using the stored
    discretization over a series of frequencies, with multiple
    computation grids based on a target number of gridpoints
    per wavelength.
    '''

    initMap = {
    #   Argument            Required    Rename as ...   Store as type
        'c':                (True,      '_c',           np.complex128),
        'freqs':            (True,      None,           list),
        'cMin':             (True,      None,           np.float64),
        'targetGPW':        (True,      None,           np.float64),
    }

    @property
    def c(self):
        'Complex wave velocity'
        if isinstance(self._c, np.ndarray):
            return self._c
        else:
            return self._c * np.ones((self.nz, self.nx), dtype=np.complex128)

    @property
    def mgHelper(self):
        'MultiGridHelper instance'

        if not hasattr(self, '_mgHelper'):
            sc = {key: self.systemConfig[key] for key in self.systemConfig}
            sc['freqs'] = self.freqs
            self._mgHelper = MultiGridHelper(sc)
        return self._mgHelper

    @property
    def spUpdates(self):
        'Updates for frequency subProblems'

        vals = []
        for i in range(len(self.freqs)):

            ds = self.mgHelper.downScalers[i]
            c = ds * self.c.ravel()

            spUpdate = {
                'freq':     self.freqs[i],
                'c':        c,
            }
            spUpdate.update(ds.scaleUpdate)
            spUpdate.update(self.addFields)
            vals.append(spUpdate)
        return vals


class ViscoMultiGridMultiFreq(ViscoMultiFreq,MultiGridMultiFreq):
    '''
    Wrapper to carry out forward-modelling using the stored
    discretization over a series of frequencies. Preserves
    causality by modelling velocity dispersion in the
    presence of a non-infinite Q model.
    '''

    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        # 'nky':          (False,     '_nky',         np.int64),
        'c':            (True,      '_c',           np.float64),
    }

    maskKeys = {'freqs', 'Q', 'freqBase'}

    @property
    def c(self):
        'Complex wave velocity'
        if isinstance(self._c, np.ndarray):
            return self._c
        else:
            return self._c * np.ones((self.nz, self.nx), dtype=np.float64)

    @property
    def spUpdates(self):
        'Updates for frequency subProblems'

        vals = []
        if self.disperseFreqs:
            for i in range(len(self.freqs)):
                freq = self.freqs[i]
                fact = 1. + (np.log(freq / self.freqBase) / (np.pi * self.Q))
                assert not self._any(fact < 0.1)

                ds = self.mgHelper.downScalers[i]

                cR = fact * self.c
                c = cR + (0.5j * cR / self.Q) # NB: + b/c of FT convention
                c = ds * c.ravel()

                spUpdate = {
                    'freq': freq,
                    'c':    c,
                }

                if isinstance(self.Q, np.ndarray):
                    Q = ds * self.Q.ravel()
                    spUpdate['Q'] = Q

                spUpdate.update(ds.scaleUpdate)
                spUpdate.update(self.addFields)
                vals.append(spUpdate)

        else:
            for i in range(len(self.freqs)):
                ds = self.mgHelper.downScalers[i]

                c = self.c.ravel() + (0.5j * self.c.ravel() / self.Q.ravel()) # NB: + b/c of FT convention
                c = ds * c

                spUpdate = {
                    'freq': self.freqs[i],
                    'c':    c,
                }

                if isinstance(self.Q, np.ndarray):
                    Q = ds * self.Q.ravel()
                    spUpdate['Q'] = Q

                spUpdate.update(ds.scaleUpdate)
                spUpdate.update(self.addFields)
                vals.append(spUpdate)

        return vals


class MultiGridHelper(BaseModelDependent,BaseSCCache):

    initMap = {
    #   Argument            Required    Rename as ...   Store as type
        'cMin':             (True,      None,           np.complex128),
        'freqs':            (True,      None,           list),
        'targetGPW':        (True,      None,           np.float64),
        'GridInterpolator': (False,     '_gi',          None),
        'maxScale':         (False,     '_maxScale',    np.float64),
        'minScale':         (False,     '_minScale',    np.float64),
    }

    @property
    def maxScale(self):
        return getattr(self, '_maxScale', 10.)

    @property
    def minScale(self):
        return getattr(self, '_minScale', 1.)

    @property
    def GridInterpolator(self):
        return getattr(self, '_gi', SplineGridInterpolator)

    @property
    def GIFilter(self):
        if not hasattr(self, '_GIFilter'):
            self._GIFilter = SCFilter(self.GridInterpolator)
        return self._GIFilter

    @property
    def scales(self):
        'Downscaling factors'
        return [np.median(((self.cMin / freq / self.dx / self.targetGPW).real, self.maxScale, self.minScale)) for freq in self.freqs]

    @property
    def downScalers(self):
        'Matrices to downscale'

        if not hasattr(self, '_downScalers'):

            scaleUpdates = [{key: self.systemConfig[key] for key in self.systemConfig} for scale in self.scales]
            for scale, sc in zip(self.scales, scaleUpdates):
                update = {
                    'scale':    scale,
                }
                sc.update(update)

            self._downScalers = [self.GridInterpolator(self.GIFilter(sc)) for sc in scaleUpdates]
        return self._downScalers

    @property
    def upScalers(self):
        'Matrices to upscale'

        if not hasattr(self, '_upScalers'):
            self._upScalers = [ds.T for ds in self.downScalers]
        return self._upScalers

