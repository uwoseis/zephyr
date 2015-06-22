import numpy as np
from IPython.parallel import Reference, interactive
from SimPEG import Survey, Problem, Mesh, Solver as SimpegSolver
from SimPEG.Parallel import RemoteInterface, SystemSolver
from SimPEG.Utils import CommonReducer
from zephyr.Survey import SurveyHelm
from zephyr.Problem import ProblemHelm
import networkx

DEFAULT_DTYPE = 'double'
DEFAULT_MPI = True
MPI_BELLWETHERS = ['PMI_SIZE', 'OMPI_UNIVERSE_SIZE']


class SeisFDFDDispatcher(object):
    """
    Base problem class for FDFD (Frequency Domain Finite Difference)
    modelling of systems for seismic imaging.
    """

    #surveyPair = Survey.BaseSurvey
    #dataPair = Survey.Data
    systemConfig = {}

    Solver = SimpegSolver
    solverOpts = {}

    def __init__(self, systemConfig, **kwargs):

        self.systemConfig = systemConfig.copy()

        hx = [(self.systemConfig['dx'], self.systemConfig['nx']-1)]
        hz = [(self.systemConfig['dz'], self.systemConfig['nz']-1)]
        self.mesh = Mesh.TensorMesh([hx, hz], '00')

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
        import zephyr.Kernel as Kernel
        from zephyr.Dispatcher import SeisFDFDDispatcher
        '''

        self.remote = RemoteInterface(systemConfig.get('profile', None), systemConfig.get('MPI', None), bootstrap=bootstrap)

        localcache = ['chunksPerWorker', 'ensembleClear']
        for key in localcache:
            if key in self.systemConfig:
                setattr(self, '_%s'%(key,), systemConfig[key])

        self.rebuildSystem()


    def _setupRemoteSystems(self, systemConfig, subConfigSettings):

        from IPython.parallel.client.remotefunction import ParallelFunction
        from SimPEG.Parallel import Endpoint
        from zephyr.Kernel import SeisFDFDKernel

        funcRef = lambda name: Reference('%s.%s'%(self.__class__.__name__, name))

        # NB: The name of the Endpoint in the remote namespace should be propagated
        #     from this function. Everything else is adaptable, to allow for changing
        #     the namespace in a single place.
        self.endpointName = 'endpoint'

        # Begin construction of Endpoint object
        endpoint = Endpoint()

        # endpoint.functions = {
        #     'forwardFromTagAccumulate':     SeisFDFDDispatcher._forwardFromTagAccumulate,     # funcRef('_forwardFromTagAccumulate'),
        #     'forwardFromTagAccumulateAll':  SeisFDFDDispatcher._forwardFromTagAccumulateAll,  # funcRef('_forwardFromTagAccumulateAll'),
        #     'backpropFromTagAccumulate':    SeisFDFDDispatcher._backpropFromTagAccumulate,    # funcRef('_backpropFromTagAccumulate'),
        #     'backpropFromTagAccumulateAll': SeisFDFDDispatcher._backpropFromTagAccumulateAll, # funcRef('_backpropFromTagAccumulateAll'),
        #     'clearFromTag':                 SeisFDFDDispatcher._clearFromTag,                 # funcRef('_clearFromTag'),
        # }

        endpoint.fieldspec = {
            'dPred':    CommonReducer,
            'dResid':   CommonReducer,
            'fWave':    CommonReducer,
            'bWave':    CommonReducer,
        }

        endpoint.systemFactory = SeisFDFDKernel#Reference('Kernel.SeisFDFDKernel')#lambda sc: SeisFDFDKernel(sc)
        endpoint.baseSystemConfig = systemConfig

        # End local construction of Endpoint object and send to workers
        self.remote[self.endpointName] = endpoint
        #self.remote.dview['%s.systemFactory'%(self.endpointName,)] = Reference('Kernel.SeisFDFDKernel')

        # Begin remote update of Endpoint object
        dview = self.remote.dview

        fnLoc = lambda fnName: '%s.functions["%s"]'%(self.endpointName, fnName)
        dview[fnLoc('forwardFromTagAccumulate')]        = self._forwardFromTagAccumulate
        dview[fnLoc('forwardFromTagAccumulateAll')]     = self._forwardFromTagAccumulateAll
        dview[fnLoc('backpropFromTagAccumulate')]       = self._backpropFromTagAccumulate
        dview[fnLoc('backpropFromTagAccumulateAll')]    = self._backpropFromTagAccumulateAll
        dview[fnLoc('clearFromTag')]                    = self._clearFromTag

        if getattr(self, '_srcs', None) is not None:
            dview['%s.srcs'%(self.endpointName,)] = self._srcs

        dview.apply_sync(Reference('%s.setupLocalFields'%(self.endpointName,)))

        setupFunction = ParallelFunction(dview, Reference('%s.setupLocalSystem'%(self.endpointName,)), dist='r', block=True).map
        rotate = lambda vec: vec[-1:] + vec[:-1]

        # TODO: This is non-optimal if there are fewer subproblems than workers
        subConfigs = self._gen25DSubConfigs(**subConfigSettings)
        parFac = systemConfig.get('parFac', 1)
        while parFac > 0:
            setupFunction(subConfigs)
            # dview.map_sync(Reference('endpoint.setupLocalSystem'), subConfigs)
            # lview.map_sync(self._setupSystem, subConfigs)
            subConfigs = rotate(subConfigs)
            parFac -= 1

        # End remote update of Endpoint object

        schedule = {
            'forward': {'solve': 'forwardFromTagAccumulateAll', 'clear': 'clearFromTag', 'reduce': ['dPred', 'fWave']},
            'backprop': {'solve': 'backpropFromTagAccumulateAll', 'clear': 'clearFromTag', 'reduce': ['bWave']},
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
                    'ifreq':    ifreq,
                    'iky':      iky,
                    'tag':      (ifreq, iky),
                })
        return result

    @staticmethod
    @interactive
    def _clearFromTag(endpoint, tag):
        return endpoint.localSystems[tag].clear()

    @staticmethod
    @interactive
    def _forwardFromTagAccumulate(endpoint, tag, isrc, **kwargs):

        locS = endpoint.localSystems
        locF = endpoint.localFields

        from IPython.parallel.error import UnmetDependency
        if not tag in locS:
            raise UnmetDependency

        key = tag[0]

        dPred = locF['dPred']
        if not key in dPred:
            dims = (len(endpoint.srcs), reduce(max, (src.nD for src in endpoint.srcs)))
            dPred[key] = np.zeros(dims, dtype=locS[tag].dtypeComplex)

        fWave = locF['fWave']
        if not key in fWave:
            dims = (len(endpoint.srcs), locS[tag].mesh.nN)
            fWave[key] = np.zeros(dims, dtype=locS[tag].dtypeComplex)

        u, d = locS[tag].forward(endpoint.srcs[isrc], dOnly=False, **kwargs)
        fWave[key][isrc,:] += u
        dPred[key][isrc,:] += d

    @staticmethod
    @interactive
    def _forwardFromTagAccumulateAll(endpoint, tag, isrcs, **kwargs):

        for isrc in isrcs:
            endpoint.functions['forwardFromTagAccumulate'](endpoint, tag, isrc, **kwargs)

    @staticmethod
    @interactive
    def _backpropFromTagAccumulate(endpoint, tag, isrc, **kwargs):

        locS = endpoint.localSystems
        locF = endpoint.localFields
        gloF = endpoint.globalFields

        from IPython.parallel.error import UnmetDependency
        if not tag in locS:
            raise UnmetDependency

        key = tag[0]

        bWave = locF['bWave']
        if not key in bWave:
            dims = (len(endpoint.srcs), locS[tag].mesh.nN)
            bWave[key] = np.zeros(dims, dtype=locS[tag].dtypeComplex)

        dResid = gloF.get('dResid', None)
        if dResid is not None and key in dResid:
            resid = dResid[key][isrc,:]
            u = locS[tag].backprop(endpoint.srcs[isrc], np.conj(resid))
            bWave[key][isrc,:] += u

    @staticmethod
    @interactive
    # @blockOnTag
    def _backpropFromTagAccumulateAll(endpoint, tag, isrcs, **kwargs):

        for isrc in isrcs:
            endpoint.functions['backpropFromTagAccumulate'](endpoint, tag, isrc, **kwargs) 

    # Fields
    def forward(self):

        if self.srcs is None:
            raise Exception('Transmitters not defined!')

        if not self.solvedF:
            self.remote.dview.apply(Reference('%s.setupLocalFields'%self.endpointName), ['fWave', 'dPred'])
            self.forwardGraph = self.systemsolver('forward', slice(len(self.srcs)))

    def backprop(self, dresid=None):

        if self.srcs is None:
            raise Exception('Transmitters not defined!')

        # if not self.dresid:
        #     raise Exception('Data residuals not defined!')

        if not self.solvedB:
            self.remote.dview.apply(Reference('%s.setupLocalFields'%self.endpointName), ['bWave'])
            self.backpropGraph = self.systemsolver('backprop', slice(len(self.srcs)))

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

    @property
    def dObs(self):
        return getattr(self, '_dobs', None)
    @dObs.setter
    def dObs(self, value):
        self._dobs = CommonReducer(value)
        self.remote.dview["%(endpoint)s.globalFields['dObs']"%{'endpoint': self.endpointName}] = self._dobs

    def _computeResidual(self):
        if not self.solvedF:
            raise Exception('Forward problem has not been solved yet!')

        if self.dObs is None:
            raise Exception('No observed data has been defined!')

        if not getattr(self, '_residualPrecomputed', False):
            # self.remote.remoteDifferenceGatherFirst('dPred', 'dObs', 'dResid')
            # #self.remote.dview.execute('dResid = CommonReducer({key: np.log(dResid[key]).real for key in dResid.keys()}')
            self.remote.e0.execute("%(endpoint)s.globalFields['dResid'] = %(endpoint)s.globalFields['dPred'] - %(endpoint)s.globalFields['dObs']"%{'endpoint': self.endpointName})
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

    @property
    def nx(self):
        return self.systemConfig['nx']

    @property
    def nz(self):
        return self.systemConfig['nz']

    @property
    def nsrc(self):
        return len(self.systemConfig['geom']['src'])
    
    @property
    def modelDims(self):
        return (self.nz, self.nx)

    @property
    def fieldDims(self):
        return (self.nsrc, self.nz, self.nx)

    @property
    def remoteFieldDims(self):
        return (self.nsrc, self.nz*self.nx)

    def spawnInterfaces(self):

        self.survey = SurveyHelm(self)
        self.problem = ProblemHelm(self)

        self.survey.pair(self.problem)

        return self.survey, self.problem

    @property
    def chunksPerWorker(self):
        return getattr(self, '_chunksPerWorker', 1)

    @property
    def ensembleClear(self):
        return getattr(self, '_ensembleClear', False)
    
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
