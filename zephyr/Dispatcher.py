import numpy as np
from IPython.parallel import Reference, interactive
from SimPEG import Survey, Problem, Mesh, Solver as SimpegSolver
from zephyr.Parallel import RemoteInterface, CommonReducer, SystemSolver
from zephyr.Survey import SurveyHelm
from zephyr.Problem import ProblemHelm
from zephyr.Source import HelmGeneral
import networkx

DEFAULT_DTYPE = 'double'
DEFAULT_MPI = True
MPI_BELLWETHERS = ['PMI_SIZE', 'OMPI_UNIVERSE_SIZE']


@interactive
def setupSystem(scu, srcObj=None):

    import os
    import zephyr.Kernel as Kernel
    from IPython.parallel.error import UnmetDependency

    global localSystem

    tag = (scu['ifreq'], scu['iky'])

    # If there is already a system to do this job on this machine, push the duplicate to another
    if tag in localSystem:
        raise UnmetDependency

    subSystemConfig = baseSystemConfig.copy()
    subSystemConfig.update(scu)

    # Set up method output caching
    if 'cacheDir' in baseSystemConfig:
        subSystemConfig['cacheDir'] = os.path.join(baseSystemConfig['cacheDir'], 'cache', '%d-%d'%tag)

    localSystem[tag] = Kernel.SeisFDFDKernel(subSystemConfig, srcObj)

    return tag

# def blockOnTag(fn):
#     def checkForSystem(*args, **kwargs):
#         from IPython.parallel.error import UnmetDependency
#         if not args[0] in localSystem:
#             raise UnmetDependency

#         return fn(*args, **kwargs)

#     return checkForSystem


@interactive
def clearFromTag(tag):
    return localSystem[tag].clear()

@interactive
def forwardFromTagBatch(tag, isrcs, **kwargs):

    from IPython.parallel import UnmetDependency
    if tag not in localSystem:
        raise UnmetDependency

    key = tag[0]
    system = localSystem[tag]
    srcObj = system._srcObj

    if not key in dPred:
        dims = (srcObj.nsrc, srcObj.nrec)
        dPred[key] = np.zeros(dims, dtype=system.dtypeComplex)

    if not key in fWave:
        dims = (srcObj.nsrc, system.mesh.nN)
        fWave[key] = np.zeros(dims, dtype=system.dtypeComplex)

    uF, d = system.forward(isrcs, None)
    fWave[key][isrcs, :] += uF.T
    dPred[key][isrcs, :] += d.T

@interactive
def backpropFromTagBatch(tag, isrcs, **kwargs):

    from IPython.parallel import UnmetDependency
    if tag not in localSystem:
        raise UnmetDependency

    key = tag[0]
    system = localSystem[tag]
    srcObj = system._srcObj

    if not key in bWave:
        dims = (srcObj.nsrc, system.mesh.nN)
        bWave[key] = np.zeros(dims, dtype=system.dtypeComplex)

    dResid = globals().get('dResid', None)
    if dResid is not None and key in dResid:
        resid = dResid[key][isrcs,:]
        uB = system.backprop(isrcs, np.conj(resid))
        bWave[key][isrcs,:] += uB.T


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

        self.remote = RemoteInterface(systemConfig.get('profile', None), systemConfig.get('MPI', None))
        dview = self.remote.dview

        code = '''
        import numpy as np
        import scipy as scipy
        import scipy.sparse
        import SimPEG
        import zephyr.Kernel as Kernel
        '''

        for command in code.strip().split('\n'):
            dview.execute(command.strip())

        localcache = ['chunksPerWorker', 'ensembleClear']
        for key in localcache:
            if key in self.systemConfig:
                setattr(self, '_%s'%(key,), systemConfig[key])

        self.rebuildSystem()


    def _getHandles(self, systemConfig, subConfigSettings):

        pclient = self.remote.pclient
        dview = self.remote.dview
        lview = self.remote.lview

        subConfigs = self._gen25DSubConfigs(**subConfigSettings)
        nsp = len(subConfigs)

        # Set up dictionary for subproblem objects and push base configuration for the system
        dview['localSystem'] = {}
        self.remote['baseSystemConfig'] = systemConfig # Faster if MPI is available
        self.remote['srcObj'] = HelmGeneral(systemConfig['geom'])
        dview['dPred'] = CommonReducer()
        dview['fWave'] = CommonReducer()
        dview['bWave'] = CommonReducer()

        dview['forwardFromTagBatch'] = forwardFromTagBatch
        dview['backpropFromTagBatch'] = backpropFromTagBatch
        dview['clearFromTag'] = clearFromTag

        dview.wait()

        schedule = {
            'forward': {'solve': Reference('forwardFromTagBatch'), 'clear': Reference('clearFromTag'), 'reduce': ['dPred', 'fWave']},
            'backprop': {'solve': Reference('backpropFromTagBatch'), 'clear': Reference('clearFromTag'), 'reduce': ['bWave']},
        }

        self.systemsolver = SystemSolver(self, schedule)

        if 'parFac' in systemConfig:
            parFac = systemConfig['parFac']
        else:
            parFac = 1

        while parFac > 0:
            tags = lview.map_sync(setupSystem, subConfigs, (Reference('srcObj') for i in xrange(len(subConfigs))))
            parFac -= 1


    def _gen25DSubConfigs(self, freqs, nky, cmin):
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
                })
        return result

    # Fields
    def forward(self):

        if not self.solvedF:
            dview = self.remote.dview
            dview['dPred'] = CommonReducer()
            dview['fWave'] = CommonReducer()
            self.forwardGraph = self.systemsolver('forward', slice(self.nsrc))

    def backprop(self):

        if not self.solvedB:
            dview = self.remote.dview
            dview['bWave'] = CommonReducer()
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
        subConfigs = self._gen25DSubConfigs(**self._subConfigSettings)
        nsp = len(subConfigs)

        #self.curModel = self.systemConfig['c'].ravel()
        self._handles = self._getHandles(self.systemConfig, self._subConfigSettings)

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

    @property
    def uF(self):
        if self.solvedF:
            return self.remote.reduce('fWave').reshape(self.fieldDims)
        else:
            return None

    @property
    def uB(self):
        if self.solvedB:
            return self.remote.reduce('bWave').reshape(self.fieldDims)
        else:
            return None

    @property
    def dPred(self):
        if self.solvedF:
            return self.remote.reduce('dPred')
        else:
            return None

    @property
    def g(self):
        if self.solvedF and self.solvedB:
            return self.remote.reduceMul('fWave', 'bWave', axis=0).reshape(self.modelDims)
        else:
            return None

    @property
    def dObs(self):
        return getattr(self, '_dobs', None)
    @dObs.setter
    def dObs(self, value):
        self._dobs = CommonReducer(value)
        self.remote['dObs'] = self._dobs

    def _computeResidual(self):
        if not self.solvedF:
            raise Exception('Forward problem has not been solved yet!')

        if self.dObs is None:
            raise Exception('No observed data has been defined!')

        if getattr(self, '_residualPrecomputed', None) is None:
            self._residualPrecomputed = False

        if not self._residualPrecomputed:
            self.remote.remoteDifferenceGatherFirst('dPred', 'dObs', 'dResid')
            #self.remote.dview.execute('dResid = CommonReducer({key: np.log(dResid[key]).real for key in dResid.keys()}')
            self._residualPrecomputed = True

    @property
    def residual(self):
        if self.solvedF:
            self._computeResidual()
            return self.remote.e0['dResid']
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
                self._misfit = self.remote.normFromDifference('dResid')
            return self._misfit
        else:
            return None

    def srcEst(self, individual=False):
        if not getattr(self, '_srcEstimated', False):
            if self.solvedF:
                self._residualPrecomputed = False

                self.remote.remoteSrcEstGatherFirst('dPred', 'dObs', 'srcTerm')
                self.remote.remoteApplySrc('dPred', 'srcTerm')

            self._srcEstimated = True

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
    def nrec(self):
        return len(self.systemConfig['geom']['rec'])
    
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
