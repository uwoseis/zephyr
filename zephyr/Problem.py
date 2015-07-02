"""
Contains numerical kernel for Seismic FDFD class
"""

import numpy as np
import numpy.linalg as la
import zephyr
from zephyr.Survey import SeisFDFD25DSurvey
import scipy
import scipy.sparse
import SimPEG
import shutil, os, errno
from IPython.parallel import require, interactive, Reference
from SimPEG.Parallel import RemoteInterface, SystemSolver
from SimPEG.Utils import CommonReducer
import networkx

DEFAULT_FREESURF_BOUNDS = [False, False, False, False]
DEFAULT_PML_SIZE = 10
DEFAULT_IREG = 4
DEFAULT_DTYPE = 'double'
NORM_EPS = 1e-10

try:
    from pymatsolver import MumpsSolver
    DEFAULT_SOLVER = MumpsSolver
except:
    DEFAULT_SOLVER = SimPEG.SolverWrapD(scipy.sparse.linalg.splu)

class SeisFDFD25DProblem(SimPEG.Problem.BaseProblem):

    surveyPair = SeisFDFD25DSurvey

    def __init__(self, systemConfig, **kwargs):

        if systemConfig.get('cache', False):
            try:
                from tempfile import mkdtemp
                from joblib import Memory
            except ImportError:
                pass
            else:
                if 'cacheDir' in systemConfig:
                    cacheDir = systemConfig['cacheDir']
                    try:
                        os.makedirs(cacheDir)
                    except OSError as e:
                        if e.errno == errno.EEXIST and os.path.isdir(cacheDir):
                            pass
                        else:
                            raise
                else:
                    cacheDir = mkdtemp()

                self._mem = Memory(cachedir=cacheDir, verbose=0)

                # Cache outputs of these methods
                self.forward = self._mem.cache(self.forward)
                self.backprop = self._mem.cache(self.backprop)

        hx = [(systemConfig['dx'], systemConfig['nx']-1)]
        hz = [(systemConfig['dz'], systemConfig['nz']-1)]
        self.mesh = SimPEG.Mesh.TensorMesh([hx, hz], '00')

        self.mesh.ireg = systemConfig.get('ireg', DEFAULT_IREG)
        self.mesh.freeSurf = systemConfig.get('freeSurf', DEFAULT_FREESURF_BOUNDS)

        initMap = {
        #   Argument        Rename to Property
            'c':            'cR',
            'Q':            None,
            'rho':          None,
            'nPML':         None,
            'freeSurf':     None,
            'freq':         '_freq',
            'ky':           '_ky',
            'kyweight':     '_kyweight',
            'Solver':       None,
            'dx':           None,
            'dz':           None,
            'dtype':        None,
        }

        for key in initMap.keys():
            if key in systemConfig:
                if initMap[key] is None:
                    setattr(self, key, systemConfig[key])
                else:
                    setattr(self, initMap[key], systemConfig[key])

        survey = systemConfig.get('survey', None)
        if survey is None:
            geom = systemConfig.get('geom', None)
            if geom is not None:
                survey = SeisFDFD25DSurvey(geom)

        if survey is not None:
            self.pair(survey)

    def __del__(self):
        if hasattr(self, '_mem'):
            self._mem.clear()
            cacheDir = self._mem.cachedir
            del self._mem
            shutil.rmtree(cacheDir)

    # Model properties

    @property
    def curModel(self):
        return self.cR.ravel()
    @curModel.setter
    def curModel(self, value):
        if value is self.curModel:
            return

        self.cR = value
        self._invalidateMatrix()

    @property
    def c(self):
        return self.cR + self.cI
    @c.setter
    def c(self, value):
        self._cR = value.real
        self._cI = value.imag
        self._invalidateMatrix()

    @property
    def rho(self):
        if getattr(self, '_rho', None) is None:
            self._rho = 310 * self.c**0.25
        return self._rho
    @rho.setter
    def rho(self, value):
        self._rho = value
        self._invalidateMatrix()

    @property
    def Q(self):
        if getattr(self, '_Q', None) is None:
            self._Q = np.inf
        return self._Q
    @Q.setter
    def Q(self, value):
        self._Q = value
        self._invalidateMatrix()

    @property
    def cR(self):
        return self._cR
    @cR.setter
    def cR(self, value):
        self._cR = value
        self._invalidateMatrix()
    
    @property
    def cI(self):
        if self.Q is np.inf:
            return 0
        else:
            return 1j * self.cR / (2*self.Q)
    @cI.setter
    def cI(self, value):
        if (value == 0).all():
            self._Q = np.inf
        else:
            self._Q = 1j * self.cR / (2*value)
        self._invalidateMatrix()

    # Modelling properties

    @property
    def nPML(self):
        if getattr(self, '_nPML', None) is None:
            self._nPML = DEFAULT_PML_SIZE
        return self._nPML
    @nPML.setter
    def nPML(self, value):
        self._nPML = value
        self._invalidateMatrix()

    @property
    def freq(self):
        return self._freq

    @property
    def ky(self):
        if getattr(self, '_ky', None) is None:
            self._ky = 0.
        return self._ky

    @property
    def kyweight(self):
        if getattr(self, '_kyweight', None) is None:
            self._kyweight = 1.
        return self._kyweight

    # Clever matrix setup properties

    @property
    def Solver(self):
        if getattr(self, '_Solver', None) is None:
            self._Solver = DEFAULT_SOLVER
        return self._Solver
    @Solver.setter
    def Solver(self, value):
        self._Solver = value

    @property
    def A(self):
        if getattr(self, '_A', None) is None:
            self._A = self._initHelmholtzNinePoint()
        return self._A

    @property
    def Ainv(self):
        if getattr(self, '_Ainv', None) is None:
            self._mfact()
        return self._Ainv

    def _invalidateMatrix(self):
        if getattr(self, '_A', None) is not None:
            del(self._A)
        if getattr(self, '_Ainv', None) is not None:
            del(self._Ainv)
        if getattr(self, '_mem', None) is not None:
            self._mem.clear()

        for prop in self.deleteTheseOnModelUpdate:
            if hasattr(self, prop):
                delattr(self, prop) 

    @property
    def dtypeReal(self):
        if self.dtype == 'float':
            return np.float32
        elif self.dtype == 'double':
            return np.float64
        else:
            raise NotImplementedError('Unknown dtype: %s'%self.dtype)

    @property
    def dtypeComplex(self):
        if self.dtype == 'float':
            return np.complex64
        elif self.dtype == 'double':
            return np.complex128
        else:
            raise NotImplementedError('Unknown dtype: %s'%self.dtype)

    @property
    def dtype(self):
        return getattr(self, '_dtype', DEFAULT_DTYPE)
    @dtype.setter
    def dtype(self, value):
        # Currently this doesn't work because all the solvers assume doubles
        # if value in ['float', 'double']:
        if value in ['double']:
            self._dtype = value
        else:
            raise NotImplementedError('Unknown dtype: %s'%value)

    @property
    def nx(self):
        return self.mesh.nNx

    @property
    def nz(self):
        return self.mesh.nNy

    @property
    def nsrc(self):
        return self.survey.nSrc
    
    @property
    def modelDims(self):
        return (self.nz, self.nx)

    @property
    def fieldDims(self):
        return (self.nsrc, self.nz, self.nx)

    @property
    def dataDims(self):
        return (self.nsrc, self.survey.nD/self.survey.nSrc)

    @property
    def remoteFieldDims(self):
        return (self.nsrc, self.nz*self.nx)

    # ------------------------------------------------------------------------
    # Matrix setup

    def _mfact(self):
        self._Ainv = self.Solver(self.A)

    def _initHelmholtzNinePoint(self):
        """
        An attempt to reproduce the finite-difference stencil and the
        general behaviour of OMEGA by Pratt et al. The stencil is a 9-point
        second-order version based on work by a number of people in the mid-90s
        including Ivan Stekl. The boundary conditions are based on the PML
        implementation by Steve Roecker in fdfdpml.f.
        """

        # Set up SimPEG mesh
        dims = (self.mesh.nNy, self.mesh.nNx)
        # mAve = self.mesh.aveN2CC

        # c = (mAve.T * self.c.ravel()).reshape(dims)
        # rho = (mAve.T * self.rho.ravel()).reshape(dims)

        c = self.c
        rho = self.rho

        # fast --> slow is x --> y --> z as Fortran

        # Set up physical properties in matrices with padding
        omega   = 2 * np.pi * self.freq 
        cPad    = np.pad(c, pad_width=1, mode='edge')
        rhoPad  = np.pad(rho, pad_width=1, mode='edge')

        aky = 2*np.pi*self.ky

        # Model parameter M
        K = ((omega**2 / cPad**2) - aky**2) / rhoPad

        # Horizontal, vertical and diagonal geometry terms
        dx  = self.mesh.hx[0]
        dz  = self.mesh.hy[0]
        dxx = dx**2
        dzz = dz**2
        dxz = dx*dz
        dd  = np.sqrt(dxz)

        # PML decay terms
        # NB: Arrays are padded later, but 'c' in these lines
        #     comes from the original (un-padded) version

        nPML    = self.nPML

        pmldx   = dx*(nPML - 1)
        pmldz   = dz*(nPML - 1)
        pmlr    = 1e-3
        pmlfx   = 3.0 * np.log(1/pmlr)/(2*pmldx**3)
        pmlfz   = 3.0 * np.log(1/pmlr)/(2*pmldz**3)

        dpmlx   = np.zeros(dims, dtype=self.dtypeComplex)
        dpmlz   = np.zeros(dims, dtype=self.dtypeComplex)
        isnx    = np.zeros(dims, dtype=self.dtypeReal)
        isnz    = np.zeros(dims, dtype=self.dtypeReal)

        # Only enable PML if the free surface isn't set

        freeSurf = self.mesh.freeSurf

        if freeSurf[0]:    
            isnz[-nPML:,:] = -1 # Top

        if freeSurf[1]:
            isnx[:,-nPML:] = -1 # Right Side

        if freeSurf[2]:
            isnz[:nPML,:] = 1 # Bottom

        if freeSurf[3]:
            isnx[:,:nPML] = 1 # Left side

        dpmlx[:,:nPML] = (np.arange(nPML, 0, -1)*dx).reshape((1,nPML))
        dpmlx[:,-nPML:] = (np.arange(1, nPML+1, 1)*dx).reshape((1,nPML))
        dnx     = pmlfx*c*dpmlx**2
        ddnx    = 2*pmlfx*c*dpmlx
        denx    = dnx + 1j*omega
        r1x     = 1j*omega / denx
        r1xsq   = r1x**2
        r2x     = isnx*r1xsq*ddnx/denx

        dpmlz[:nPML,:] = (np.arange(nPML, 0, -1)*dz).reshape((nPML,1))
        dpmlz[-nPML:,:] = (np.arange(1, nPML+1, 1)*dz).reshape((nPML,1))
        dnz     = pmlfz*c*dpmlz**2
        ddnz    = 2*pmlfz*c*dpmlz
        denz    = dnz + 1j*omega
        r1z     = 1j*omega / denz
        r1zsq   = r1z**2
        r2z     = isnz*r1zsq*ddnz/denz

        # Visual key for finite-difference terms
        # (per Pratt and Worthington, 1990)
        #
        #   This         Original
        # AF FF CF  vs.  AD DD CD
        # AA BE CC  vs.  AA BE CC
        # AD DD CD  vs.  AF FF CF

        # Set of keys to index the dictionaries
        keys = ['AD', 'DD', 'CD', 'AA', 'BE', 'CC', 'AF', 'FF', 'CF']

        # Diagonal offsets for the sparse matrix formation
        offsets = {
            'AD':   (-1) * dims[1] + (-1), 
            'DD':   (-1) * dims[1] + ( 0),
            'CD':   (-1) * dims[1] + (+1),
            'AA':   ( 0) * dims[1] + (-1),
            'BE':   ( 0) * dims[1] + ( 0),
            'CC':   ( 0) * dims[1] + (+1),
            'AF':   (+1) * dims[1] + (-1),
            'FF':   (+1) * dims[1] + ( 0),
            'CF':   (+1) * dims[1] + (+1),
        }

        # Buoyancies
        bMM = 1. / rhoPad[0:-2,0:-2] # bottom left
        bME = 1. / rhoPad[0:-2,1:-1] # bottom centre
        bMP = 1. / rhoPad[0:-2,2:  ] # bottom right
        bEM = 1. / rhoPad[1:-1,0:-2] # middle left
        bEE = 1. / rhoPad[1:-1,1:-1] # middle centre
        bEP = 1. / rhoPad[1:-1,2:  ] # middle right
        bPM = 1. / rhoPad[2:  ,0:-2] # top    left
        bPE = 1. / rhoPad[2:  ,1:-1] # top    centre
        bPP = 1. / rhoPad[2:  ,2:  ] # top    right

        # Initialize averaged buoyancies on most of the grid
        bMM = (bEE + bMM) / 2 # a2
        bME = (bEE + bME) / 2 # d1
        bMP = (bEE + bMP) / 2 # d2
        bEM = (bEE + bEM) / 2 # a1
        # ... middle
        bEP = (bEE + bEP) / 2 # c1
        bPM = (bEE + bPM) / 2 # f2
        bPE = (bEE + bPE) / 2 # f1
        bPP = (bEE + bPP) / 2 # c2

        # Reset the buoyancies on the outside edges
        # bMM[ 0, :] = bEE[ 0, :]
        # bMM[ :, 0] = bEE[ :, 0]
        # bME[ 0, :] = bEE[ 0, :]
        # bMP[ 0, :] = bEE[ 0, :]
        # bMP[ :,-1] = bEE[ :,-1]
        # bEM[ :, 0] = bEE[ :, 0]
        # bEP[ :,-1] = bEE[ :,-1]
        # bPM[-1, :] = bEE[-1, :]
        # bPM[ :, 0] = bEE[ :, 0]
        # bPE[-1, :] = bEE[-1, :]
        # bPP[-1, :] = bEE[-1, :]
        # bPP[ :,-1] = bEE[ :,-1]

        # K = omega^2/(c^2 . rho)
        kMM = K[0:-2,0:-2] # bottom left
        kME = K[0:-2,1:-1] # bottom centre
        kMP = K[0:-2,2:  ] # bottom centre
        kEM = K[1:-1,0:-2] # middle left
        kEE = K[1:-1,1:-1] # middle centre
        kEP = K[1:-1,2:  ] # middle right
        kPM = K[2:  ,0:-2] # top    left
        kPE = K[2:  ,1:-1] # top    centre
        kPP = K[2:  ,2:  ] # top    right

        # 9-point fd star
        acoef   = 0.5461
        bcoef   = 0.4539
        ccoef   = 0.6248
        dcoef   = 0.09381
        ecoef   = 0.000001297

        # 5-point fd star
        # acoef = 1.0
        # bcoef = 0.0
        # ecoef = 0.0

        # NB: bPM and bMP here are switched relative to S. Roecker's version
        #     in OMEGA. This is because the labelling herein is always ?ZX.

        diagonals = {
            'AD':   ecoef*kMM
                    + bcoef*bMM*((r1zsq+r1xsq)/(4*dxz) - (r2z+r2x)/(4*dd)),
            'DD':   dcoef*kME
                    + acoef*bME*(r1zsq/dz - r2z/2)/dz
                    + bcoef*(r1zsq-r1xsq)*(bMP+bMM)/(4*dxz),
            'CD':   ecoef*kMP
                    + bcoef*bMP*((r1zsq+r1xsq)/(4*dxz) - (r2z+r2x)/(4*dd)),
            'AA':   dcoef*kEM
                    + acoef*bEM*(r1xsq/dx - r2x/2)/dx
                    + bcoef*(r1xsq-r1zsq)*(bPM+bMM)/(4*dxz),
            'BE':   ccoef*kEE
                    + acoef*(r2x*(bEM-bEP)/(2*dx) + r2z*(bME-bPE)/(2*dz) - r1xsq*(bEM+bEP)/dxx - r1zsq*(bME+bPE)/dzz)
                    + bcoef*(((r2x+r2z)*(bMM-bPP) + (r2z-r2x)*(bMP-bPM))/(4*dd) - (r1xsq+r1zsq)*(bMM+bPP+bPM+bMP)/(4*dxz)),
            'CC':   dcoef*kEP
                    + acoef*bEP*(r1xsq/dx + r2x/2)/dx
                    + bcoef*(r1xsq-r1zsq)*(bMP+bPP)/(4*dxz),
            'AF':   ecoef*kPM
                    + bcoef*bPM*((r1zsq+r1xsq)/(4*dxz) - (r2z+r2x)/(4*dd)),
            'FF':   dcoef*kPE
                    + acoef*bPE*(r1zsq/dz - r2z/2)/dz
                    + bcoef*(r1zsq-r1xsq)*(bPM+bPP)/(4*dxz),
            'CF':   ecoef*kPP
                    + bcoef*bPP*((r1zsq+r1xsq)/(4*dxz) - (r2z+r2x)/(4*dd)),
        }

        diagonals['AD'] = diagonals['AD'].ravel()[dims[1]+1:          ]
        diagonals['DD'] = diagonals['DD'].ravel()[dims[1]  :          ]
        diagonals['CD'] = diagonals['CD'].ravel()[dims[1]-1:          ]
        diagonals['AA'] = diagonals['AA'].ravel()[        1:          ]
        diagonals['BE'] = diagonals['BE'].ravel()[         :          ]
        diagonals['CC'] = diagonals['CC'].ravel()[         :-1        ]
        diagonals['AF'] = diagonals['AF'].ravel()[         :-dims[1]+1]
        diagonals['FF'] = diagonals['FF'].ravel()[         :-dims[1]  ]
        diagonals['CF'] = diagonals['CF'].ravel()[         :-dims[1]-1]

        # self._setupBoundary(diagonals, freeSurf)
        if any(freeSurf):
            raise NotImplementedError('Free surface not implemented!')

        # for key in diagonals.keys():
        #     print('%s:\t%d\t%d'%(key, diagonals[key].size, offsets[key]))

        diagonals = [diagonals[key] for key in keys]
        offsets = [offsets[key] for key in keys]

        A = scipy.sparse.diags(diagonals, offsets, shape=(self.mesh.nN, self.mesh.nN), format='csr', dtype=self.dtypeComplex)#, shape=(self.mesh.nN, self.mesh.nN))#, self.mesh.nN, self.mesh.nN, format='csr')

        return A

    # def _setupBoundary(self, diagonals, freeSurf):
    #     """
    #     Function to set up boundary regions for the Seismic FDFD problem
    #     using the 9-point finite-difference stencil from OMEGA/FULLWV.
    #     """

    #     keys = diagonals.keys()
    #     pickDiag = lambda x: -1. if freeSurf[x] else 1.

    #     # Left
    #     for key in keys:
    #         if key is 'BE':
    #             diagonals[key][:,0] = pickDiag(3)
    #         else:
    #             diagonals[key][:,0] = 0.

    #     # Right
    #     for key in keys:
    #         if key is 'BE':
    #             diagonals[key][:,-1] = pickDiag(1)
    #         else:
    #             diagonals[key][:,-1] = 0.

    #     # Bottom
    #     for key in keys:
    #         if key is 'BE':
    #             diagonals[key][0,:] = pickDiag(2)
    #         else:
    #             diagonals[key][0,:] = 0.

    #     # Top
    #     for key in keys:
    #         if key is 'BE':
    #             diagonals[key][-1,:] = pickDiag(0)
    #         else:
    #             diagonals[key][-1,:] = 0.

    @staticmethod
    def _densify(inmat):

        if getattr(inmat, 'todense', None) is not None:
            inmat = inmat.todense()

        return np.array(inmat)

    # ------------------------------------------------------------------------
    # Externally-callable functions

    def clear(self):
        self._invalidateMatrix()
    
    def forward(self, isrc=slice(None), coeffs=None):

        q = self.kyweight * self.survey.getSrcP(coeffs)[isrc].T
        uF = self.Ainv * self._densify(q)

        indices = isrc.indices(self.survey.nSrc)

        mapf = lambda P, si: P*uF[:,si]
        if self.ky != 0.:
            d = np.array(map(mapf, self.survey.getRecPAll(isrc, None, self.ky), xrange(*indices)))
        else:
            d = self.survey.getRecP(0, None, 0.) * uF

        subFieldDims = (indices[1] - indices[0], self.remoteFieldDims[1])
        subDataDims =  (indices[1] - indices[0], self.dataDims[1])
        return uF.T.reshape(subFieldDims), d.T.reshape(subDataDims)

    def backprop(self, isrc=slice(None), dresid=None):

        qr = self.kyweight * np.array(np.concatenate([P.sum(axis=0) for P in self.survey.getRecPAll(isrc, dresid, self.ky)])).T
        uB = self.Ainv * qr

        indices = isrc.indices(self.survey.nSrc)
        subFieldDims = (indices[1] - indices[0], self.remoteFieldDims[1])
        return uB.T.reshape(subFieldDims)

    def pair(self, d):
        """Bind a survey to this problem instance using pointers."""
        assert isinstance(d, self.surveyPair), "Data object must be an instance of a %s class."%(self.surveyPair.__name__)
        if not d.ispaired:
            d._prob = self 
        self._survey = d
    

class SeisFDFD25DParallelProblem(SimPEG.Problem.BaseProblem):
    """
    Base problem class for FDFD (Frequency Domain Finite Difference)
    modelling of systems for seismic imaging.
    """

    surveyPair = SeisFDFD25DSurvey
    #dataPair = Survey.Data
    systemConfig = {}

    def __init__(self, systemConfig, **kwargs):

        self.systemConfig = systemConfig.copy()

        hx = [(self.systemConfig['dx'], self.systemConfig['nx']-1)]
        hz = [(self.systemConfig['dz'], self.systemConfig['nz']-1)]
        self.mesh = SimPEG.Mesh.TensorMesh([hx, hz], '00')

        SimPEG.Problem.BaseProblem.__init__(self, self.mesh, **kwargs)

        # NB: Remember to set up something to do geometry conversion
        #     from origin geometry to local geometry. Functions that
        #     wrap the geometry vectors are probably easiest.

        splitkeys = ['freqs', 'nky']

        subConfigSettings = {}
        for key in splitkeys:
            value = self.systemConfig.pop(key, None)
            if value is not None:
                subConfigSettings[key] = value

        self._subConfigSettings = subConfigSettings

        bootstrap = '''
        import numpy as np
        import scipy as scipy
        import scipy.sparse
        import SimPEG
        from zephyr.Problem import SeisFDFD25DProblem
        from zephyr.Survey import SeisFDFD25DSurvey
        '''

        self.remote = RemoteInterface(systemConfig.get('profile', None), systemConfig.get('MPI', None), bootstrap=bootstrap)

        localcache = ['chunksPerWorker', 'ensembleClear', 'estimateSource']
        for key in localcache:
            if key in self.systemConfig:
                setattr(self, '_%s'%(key,), systemConfig[key])

        self.rebuildSystem()


    def _setupRemoteSystems(self, systemConfig, subConfigSettings):

        from IPython.parallel.client.remotefunction import ParallelFunction
        from SimPEG.Parallel import Endpoint
        from zephyr.Problem import SeisFDFD25DProblem

        funcRef = lambda name: Reference('%s.%s'%(self.__class__.__name__, name))

        # NB: The name of the Endpoint in the remote namespace should be propagated
        #     from this function. Everything else is adaptable, to allow for changing
        #     the namespace in a single place.
        self.endpointName = 'endpoint'

        # Begin construction of Endpoint object
        endpoint = Endpoint()

        endpoint.fieldspec = {
            'dPred':    CommonReducer,
            'fWave':    CommonReducer,
            'bWave':    CommonReducer,
        }

        endpoint.problemFactory = SeisFDFD25DProblem
        endpoint.surveyFactory = SeisFDFD25DSurvey
        endpoint.baseSystemConfig = systemConfig

        # End local construction of Endpoint object and send to workers
        self.remote[self.endpointName] = endpoint
        #self.remote.dview['%s.systemFactory'%(self.endpointName,)] = Reference('Kernel.SeisFDFDKernel')

        # Begin remote update of Endpoint object
        dview = self.remote.dview

        fnLoc = lambda fnName: '%s.functions["%s"]'%(self.endpointName, fnName)
        dview[fnLoc('forwardFromTagAccumulate')]        = self._forwardFromTagAccumulate
        dview[fnLoc('backpropFromTagAccumulate')]       = self._backpropFromTagAccumulate
        dview[fnLoc('clearFromTag')]                    = self._clearFromTag

        if getattr(self, '_srcs', None) is not None:
            dview['%s.srcs'%(self.endpointName,)] = self._srcs

        dview.apply_sync(Reference('%s.setupLocalFields'%(self.endpointName,)))

        surveySubConfigs = {ifreq: {} for ifreq in xrange(len(subConfigSettings['freqs']))}
        dview.apply_sync(Reference('%s.setupLocalSurveys'%(self.endpointName,)), surveySubConfigs)

        problemSetupFunction = ParallelFunction(dview, Reference('%s.setupLocalProblem'%(self.endpointName,)), dist='r', block=True).map
        rotate = lambda vec: vec[-1:] + vec[:-1]

        # TODO: This is non-optimal if there are fewer subproblems than workers
        subConfigs = self._gen25DSubConfigs(**subConfigSettings)
        parFac = systemConfig.get('parFac', 1)
        while parFac > 0:
            problemSetupFunction(subConfigs)
            subConfigs = rotate(subConfigs)
            parFac -= 1

        # End remote update of Endpoint object

        schedule = {
            'forward': {'solve': 'forwardFromTagAccumulate', 'clear': 'clearFromTag', 'reduce': ['dPred', 'fWave']},
            'backprop': {'solve': 'backpropFromTagAccumulate', 'clear': 'clearFromTag', 'reduce': ['bWave']},
        }

        self.systemsolver = SystemSolver(self, self.endpointName, schedule)

    @staticmethod
    def _gen25DSubConfigs(freqs, nky, cmin):
        result = []
        weightfac = 1./(2*nky - 1) if nky > 1 else 1.# alternatively, 1/dky
        for ifreq, freq in enumerate(freqs):
            k_c = freq / cmin
            dky = k_c / (nky - 1) if nky > 1 else 0.
            for iky, ky in enumerate(np.linspace(0, k_c, nky)):
                result.append({
                    'freq':     freq,
                    'ky':       ky,
                    'kyweight': 2*weightfac if ky != 0 else weightfac,
                    'isub':     ifreq,
                    'tag':      (ifreq, iky),
                })
        return result

    @staticmethod
    @interactive
    def _clearFromTag(endpoint, tag):
        return endpoint.localProblems[tag].clear()

    @staticmethod
    @interactive
    def _forwardFromTagAccumulate(endpoint, tag, isrcs, **kwargs):

        locP = endpoint.localProblems
        locF = endpoint.localFields

        from IPython.parallel.error import UnmetDependency
        if not tag in locP:
            raise UnmetDependency

        key = tag[0]

        dPred = locF['dPred']
        if not key in dPred:
            dims = locP[tag].dataDims
            dPred[key] = np.zeros(dims, dtype=locP[tag].dtypeComplex)

        fWave = locF['fWave']
        if not key in fWave:
            dims = locP[tag].remoteFieldDims
            fWave[key] = np.zeros(dims, dtype=locP[tag].dtypeComplex)

        u, d = locP[tag].forward(isrcs, **kwargs)
        fWave[key][isrcs,:] += u
        dPred[key][isrcs,:] += d

    @staticmethod
    @interactive
    def _backpropFromTagAccumulate(endpoint, tag, isrcs, **kwargs):

        locP = endpoint.localProblems
        locF = endpoint.localFields
        gloF = endpoint.globalFields

        from IPython.parallel.error import UnmetDependency
        if not tag in locP:
            raise UnmetDependency

        key = tag[0]

        bWave = locF['bWave']
        if not key in bWave:
            dims = locP[tag].remoteFieldDims
            bWave[key] = np.zeros(dims, dtype=locP[tag].dtypeComplex)

        dResid = gloF.get('dResid', None)
        if dResid is not None and key in dResid:
            resid = dResid[key][isrcs,:]
            u = locP[tag].backprop(isrcs, np.conj(resid)) # TODO: Check if this should be conj...?
            bWave[key][isrcs,:] += u

    # Fields
    def forward(self):

        # if self.srcs is None:
        #     raise Exception('Sources not defined!')

        if not self.solvedF:
            self.remote.dview.apply(Reference('%s.setupLocalFields'%self.endpointName), ['fWave', 'dPred'])
            self.forwardGraph = self.systemsolver('forward', slice(self.nsrc))

    def backprop(self, dresid=None):

        # if self.srcs is None:
        #     raise Exception('Sources not defined!')

        # if not self.dresid:
        #     raise Exception('Data residuals not defined!')

        if not self.solvedB:
            self.remote.dview.apply(Reference('%s.setupLocalFields'%self.endpointName), ['bWave'])
            self.backpropGraph = self.systemsolver('backprop', slice(self.nsrc))

    def rebuildSystem(self, c = None):
        if c is not None:
            self.systemConfig['c'] = c
            self.rebuildSystem()
            return

        if hasattr(self, 'forwardGraph'):
            del self.forwardGraph

        if hasattr(self, 'backpropGraph'):
            del self.backpropGraph

        self._solvedF = False
        self._solvedB = False
        self._residualPrecomputed = False
        self._srcEstimated = False
        self._misfit = None

        self._subConfigSettings['cmin'] = self.systemConfig['c'].min()

        #self.curModel = self.systemConfig['c'].ravel()
        self._handles = self._setupRemoteSystems(self.systemConfig, self._subConfigSettings)

    @property
    def srcs(self):
        if getattr(self, '_srcs', None) is None:
            self._srcs = None
        return self._srcs
    @srcs.setter
    def srcs(self, value):
        self._srcs = value
        self.rebuildSystem()
        self.remote['%s.srcs'%self.endpointName] = self._srcs

    @property
    def solvedF(self):
        if getattr(self, '_solvedF', None) is None:
            self._solvedF = False

        if hasattr(self, 'forwardGraph'):
            self.systemsolver.wait(self.forwardGraph)
            self._solvedF = True

        return self._solvedF

    @property
    def solvedB(self):
        if getattr(self, '_solvedB', None) is None:
            self._solvedB = False

        if hasattr(self, 'backpropGraph'):
            self.systemsolver.wait(self.backpropGraph)
            self._solvedB = True

        return self._solvedB

    def _getGlobalField(self, fieldName):
        return self.remote.e0['%s.globalFields["%s"]'%(self.endpointName, fieldName)]

    @property
    def uF(self):
        if self.solvedF:
            return self._getGlobalField('fWave').reshape(self.fieldDims)
        else:
            return None

    @property
    def uB(self):
        if self.solvedB:
            return self._getGlobalField('bWave').reshape(self.fieldDims)
        else:
            return None

    @property
    def dPred(self):
        if self.solvedF:
            return self._getGlobalField('dPred')
        else:
            return None

    @property
    def g(self):
        if self.solvedF and self.solvedB:
            return self.remote.remoteMulE0(
                "%(endpoint)s.globalFields['fWave']"%{'endpoint': self.endpointName},
                "%(endpoint)s.globalFields['bWave']"%{'endpoint': self.endpointName},
                axis=0).reshape(self.modelDims)
        else:
            return None

    def Jtvec(self, m, v, u=None):

        if not (m == self.curModel):
            self.curModel = m

        if u is not None:
            raise Exception('Providing a wavefield is not supported!')
            self.remote.dview["%(endpoint)s.globalFields['dPred']"%{'endpoint': self.endpointName}] = CommonReducer(u)
            self.solvedF = True
        else:    
            self.forward()

        if self.solvedF:
            pass

        if v is not None:
            self.remote.dview["%(endpoint)s.globalFields['dResid']"%{'endpoint': self.endpointName}] = CommonReducer(v)
            self._residualPrecomputed = True
        else:
            self._computeResidual()

        self.backprop()

        return self.g.ravel()

    @property
    def dObs(self):
        return getattr(self, '_dobs', None)
    @dObs.setter
    def dObs(self, value):
        self._dobs = CommonReducer(value)
        self.remote.e0["%(endpoint)s.globalFields['dObs']"%{'endpoint': self.endpointName}] = self._dobs

    def _computeResidual(self):
        if not self.solvedF:
            raise Exception('Forward problem has not been solved yet!')
            pass

        if self.dObs is None:
            raise Exception('No observed data has been defined!')

        if not getattr(self, '_residualPrecomputed', False):

            if self.estimateSource:
                self.srcEst()
            else:
                code = "if 'srcTerm' not in %(endpoint)s.globalFields:\n    %(endpoint)s.globalFields['srcTerm'] = {key: 1. for key in %(endpoint)s.globalFields['dPred']}"
                self.remote.e0.execute(code%{'endpoint': self.endpointName})
            # self.remote.remoteDifferenceGatherFirst('dPred', 'dObs', 'dResid')
            # #self.remote.dview.execute('dResid = CommonReducer({key: np.log(dResid[key]).real for key in dResid.keys()}')
            code = "%(endpoint)s.globalFields['dResid'] = %(endpoint)s.globalFields['srcTerm']*%(endpoint)s.globalFields['dPred'] - %(endpoint)s.globalFields['dObs']"
            self.remote.e0.execute(code%{'endpoint': self.endpointName})
            parallelcode = "if rank != %(root)d:\n    %(endpoint)s.globalFields['dResid'] = None\n%(endpoint)s.globalFields['dResid'] = comm.bcast(%(endpoint)s.globalFields['dResid'], 0)"
            self.remote.dview.execute(parallelcode%{'endpoint': self.endpointName, 'root': 0})
            self._residualPrecomputed = True

    @property
    def residual(self):
        if self.solvedF:
            self._computeResidual()
            return self.remote.e0["%(endpoint)s.globalFields['dResid']"%{'endpoint': self.endpointName}]
        else:
            return None
    # A day may come when it may be useful to set this, or to set dPred; but it is not this day!
    # @residual.setter
    # def residual(self, value):
    #     self.remote['dResid'] = CommonReducer(value)

    @property
    def misfit(self):
        if self.solvedF:
            if getattr(self, '_misfit', None) is None:
                self._computeResidual()
                self._misfit = self.remote.normFromDifference("%(endpoint)s.globalFields['dResid']"%{'endpoint': self.endpointName})
            return self._misfit
        else:
            return None

    def srcEst(self, individual=False):
        if not getattr(self, '_srcEstimated', False):
            if self.solvedF:
                self._residualPrecomputed = False

                dPred = "%(endpoint)s.globalFields['dPred']"%{'endpoint': self.endpointName}
                dObs = "%(endpoint)s.globalFields['dObs']"%{'endpoint': self.endpointName}
                srcTerm = "%(endpoint)s.globalFields['srcTerm']"%{'endpoint': self.endpointName}

                self.remote.remoteSrcEstGatherFirst(srcTerm, dPred, dObs, individual)

                self._srcEstimated = True

    @property
    def nx(self):
        return self.mesh.nNx

    @property
    def nz(self):
        return self.mesh.nNy

    @property
    def nsrc(self):
        return self.survey.nSrc
    
    @property
    def modelDims(self):
        return (self.nz, self.nx)

    @property
    def fieldDims(self):
        return (self.nsrc, self.nz, self.nx)

    @property
    def dataDims(self):
        return (self.nsrc, self.survey.nD/self.survey.nSrc)
    
    @property
    def remoteFieldDims(self):
        return (self.nsrc, self.nz*self.nx)

    # def spawnInterfaces(self):

    #     self.survey = SurveyHelm(self)
    #     self.problem = ProblemHelm(self)

    #     self.survey.pair(self.problem)

    #     return self.survey, self.problem

    @property
    def chunksPerWorker(self):
        return getattr(self, '_chunksPerWorker', 1)

    @property
    def ensembleClear(self):
        return getattr(self, '_ensembleClear', False)

    @property
    def estimateSource(self):
        return getattr(self, '_estimateSource', True)

    @property
    def curModel(self):
        return self.systemConfig['c'].ravel()
    @curModel.setter
    def curModel(self, value):
        self.rebuildSystem(value.reshape(self.modelDims))
    
    # def fields(self, c):

    #     self._rebuildSystem(c)

    #     # F = FieldsSeisFDFD(self.mesh, self.survey)

    #     # for freq in self.survey.freqs:
    #     #     A = self._initHelmholtzNinePoint(freq)
    #     #     q = self.survey.getTransmitters(freq)
    #     #     Ainv = self.Solver(A, **self.solverOpts)
    #     #     sol = Ainv * q
    #     #     F[q, 'u'] = sol

    #     return F

    # def Jvec(self, m, v, u=None):
    #     pass

    # def Jtvec(self, m, v, u=None):
    #     pass
