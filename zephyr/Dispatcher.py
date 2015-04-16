import numpy as np
from IPython.parallel import Reference, interactive
from SimPEG import Survey, Problem, Mesh, Solver as SimpegSolver
from zephyr.Parallel import RemoteInterface, commonReducer
from zephyr.Survey import SurveyHelm
from zephyr.Problem import ProblemHelm
import networkx

DEFAULT_MPI = True
MPI_BELLWETHERS = ['PMI_SIZE', 'OMPI_UNIVERSE_SIZE']


@interactive
def setupSystem(scu):

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

    localSystem[tag] = Kernel.SeisFDFDKernel(subSystemConfig)

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
# @blockOnTag
def forwardFromTagAccumulate(tag, isrc, **kwargs):

    from IPython.parallel.error import UnmetDependency
    if not tag in localSystem:
        raise UnmetDependency

    key = tag[0]

    if not key in dPred:
        dims = (len(txs), reduce(max, (tx.nD for tx in txs)))
        dPred[key] = np.zeros(dims, dtype=np.complex128)

    if not key in fWave:
        dims = (len(txs), localSystem[tag].mesh.nN)
        fWave[key] = np.zeros(dims, dtype=np.complex128)

    u, d = localSystem[tag].forward(txs[isrc], dOnly=False, **kwargs)
    fWave[key][isrc,:] += u
    dPred[key][isrc,:] += d

@interactive
# @blockOnTag
def forwardFromTagAccumulateAll(tag, isrcs, **kwargs):

    for isrc in isrcs:
        forwardFromTagAccumulate(tag, isrc, **kwargs)

@interactive
# @blockOnTag
def backpropFromTagAccumulate(tag, isrc, **kwargs):

    from IPython.parallel.error import UnmetDependency
    if not tag in localSystem:
        raise UnmetDependency

    key = tag[0]

    if not key in bWave:
        dims = (len(txs), localSystem[tag].mesh.nN)
        bWave[key] = np.zeros(dims, dtype=np.complex128)

    if key in dResid:
        resid = dResid[key][isrc,:]
    else:
        resid = 0.

    u = localSystem[tag].backprop(txs[isrc], np.conj(resid))
    bWave[key][isrc,:] += u

@interactive
# @blockOnTag
def backpropFromTagAccumulateAll(tag, isrcs, **kwargs):

    for isrc in isrcs:
        backpropFromTagAccumulate(tag, isrc, **kwargs) 

@interactive
def hasSystem(tag):
    global localSystem
    return tag in localSystem

@interactive
def hasSystemRank(tag, wid):
    global localSystem
    global rank
    return (tag in localSystem) and (rank == wid)

def getChunks(problems, chunks=1):
    nproblems = len(problems)
    return (problems[i*nproblems // chunks: (i+1)*nproblems // chunks] for i in range(chunks))

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

        self._remote = RemoteInterface(systemConfig.get('profile', None), systemConfig.get('MPI', None))
        dview = self._remote.dview

        code = '''
        import numpy as np
        import scipy as scipy
        import scipy.sparse
        import SimPEG
        import zephyr.Kernel as Kernel
        '''

        for command in code.strip().split('\n'):
            dview.execute(command.strip())

        self.rebuildSystem()


    def _getHandles(self, systemConfig, subConfigSettings):

        pclient = self._remote.pclient
        dview = self._remote.dview
        lview = self._remote.lview

        subConfigs = self._gen25DSubConfigs(**subConfigSettings)
        nsp = len(subConfigs)

        # Set up dictionary for subproblem objects and push base configuration for the system
        dview['localSystem'] = {}
        self._remote['baseSystemConfig'] = systemConfig # Faster if MPI is available
        dview['dPred'] = commonReducer()
        dview['fWave'] = commonReducer()
        dview['bWave'] = commonReducer()

        dview['forwardFromTagAccumulate'] = forwardFromTagAccumulate
        dview['forwardFromTagAccumulateAll'] = forwardFromTagAccumulateAll
        dview['backpropFromTagAccumulate'] = backpropFromTagAccumulate
        dview['backpropFromTagAccumulateAll'] = backpropFromTagAccumulateAll
        dview['clearFromTag'] = clearFromTag

        dview.wait()

        if 'parFac' in systemConfig:
            parFac = systemConfig['parFac']
        else:
            parFac = 1

        while parFac > 0:
            tags = lview.map_sync(setupSystem, subConfigs)
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

        if self.txs is None:
            raise Exception('Transmitters not defined!')

        if not self.solvedF:
            dview = self._remote.dview
            dview['dPred'] = commonReducer()
            dview['fWave'] = commonReducer()
            self.forwardGraph = self._systemSolve(Reference('forwardFromTagAccumulateAll'), slice(len(self.txs)))

    def backprop(self, dresid=None):

        if self.txs is None:
            raise Exception('Transmitters not defined!')

        # if not self.dresid:
        #     raise Exception('Data residuals not defined!')

        if not self.solvedB:
            dview = self._remote.dview
            dview['bWave'] = commonReducer()
            self.backpropGraph = self._systemSolve(Reference('backpropFromTagAccumulateAll'), slice(len(self.txs)))

    def _wait(self, G):
        self._remote.lview.wait((G.node[wn]['job'] for wn in (G.predecessors(tn)[0] for tn in G.predecessors('End'))))

    def _systemSolve(self, fnRef, isrcs, clearRef=Reference('clearFromTag'), **kwargs):

        dview = self._remote.dview
        lview = self._remote.lview

        chunksPerWorker = self.systemConfig.get('chunksPerWorker', 1)

        G = networkx.DiGraph()

        mainNode = 'Beginning'
        G.add_node(mainNode)

        # Parse sources
        nsrc = self.nsrc
        if isrcs is None:
            isrcslist = range(nsrc)

        elif isinstance(isrcs, slice):
            isrcslist = range(isrcs.start or 0, isrcs.stop or nsrc, isrcs.step or 1)

        else:
            try:
                _ = isrcs[0]
                isrcslist = isrcs
            except TypeError:
                isrcslist = [isrcs]

        systemsOnWorkers = dview['localSystem.keys()']
        ids = dview['rank']
        tags = set()
        for ltags in systemsOnWorkers:
            tags = tags.union(set(ltags))

        endNodes = {}
        tailNodes = []

        for tag in tags:

            tagNode = 'Head: %d, %d'%tag
            G.add_edge(mainNode, tagNode)

            relIDs = []
            for i in xrange(len(ids)):

                systems = systemsOnWorkers[i]
                rank = ids[i]

                if tag in systems:
                    relIDs.append(i)

            systemJobs = []
            endNodes[tag] = []
            systemNodes = []

            with lview.temp_flags(block=False):
                iworks = 0
                for work in getChunks(isrcslist, int(round(chunksPerWorker*len(relIDs)))):
                    if work:
                        job = lview.apply(fnRef, tag, work, **kwargs)
                        systemJobs.append(job)
                        label = 'Compute: %d, %d, %d'%(tag[0], tag[1], iworks)
                        systemNodes.append(label)
                        G.add_node(label, job=job)
                        G.add_edge(tagNode, label)
                        iworks += 1

            if self.systemConfig.get('ensembleClear', False): # True for ensemble ending, False for individual ending
                tagNode = 'Wrap: %d, %d'%tag
                for label in systemNodes:
                    G.add_edge(label, tagNode)

                for i in relIDs:

                    rank = ids[i]

                    with lview.temp_flags(block=False, after=systemJobs):
                        job = lview.apply(depend(hasSystemRank, tag, rank)(clearRef), tag)
                        label = 'Wrap: %d, %d, %d'%(tag[0],tag[1], i)
                        G.add_node(label, job=job)
                        endNodes[tag].append(label)
                        G.add_edge(tagNode, label)
            else:

                for i, sjob in enumerate(systemJobs):
                    with lview.temp_flags(block=False, follow=sjob):
                        job = lview.apply(clearRef, tag)
                        label = 'Wrap: %d, %d, %d'%(tag[0],tag[1],i)
                        G.add_node(label, job=job)
                        endNodes[tag].append(label)
                        G.add_edge(systemNodes[i], label)

            tagNode = 'Tail: %d, %d'%tag
            for label in endNodes[tag]:
                G.add_edge(label, tagNode)
            tailNodes.append(tagNode)

        endNode = 'End'
        for node in tailNodes:
            G.add_edge(node, endNode)

        return G

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
        subConfigs = self._gen25DSubConfigs(**self._subConfigSettings)
        nsp = len(subConfigs)

        #self.curModel = self.systemConfig['c'].ravel()
        self._handles = self._getHandles(self.systemConfig, self._subConfigSettings)

    @property
    def txs(self):
        if getattr(self, '_txs', None) is None:
            self._txs = None
        return self._txs
    @txs.setter
    def txs(self, value):
        self._txs = value
        self.rebuildSystem()
        self._remote['txs'] = self._txs

    @property
    def solvedF(self):
        if getattr(self, '_solvedF', None) is None:
            self._solvedF = False

        if hasattr(self, 'forwardGraph'):
            self._wait(self.forwardGraph)
            self._solvedF = True

        return self._solvedF

    @property
    def solvedB(self):
        if getattr(self, '_solvedB', None) is None:
            self._solvedB = False

        if hasattr(self, 'backpropGraph'):
            self._wait(self.backpropGraph)
            self._solvedB = True

        return self._solvedB

    @property
    def uF(self):
        if self.solvedF:
            return self._remote.reduce('fWave').reshape(self.fieldDims)
        else:
            return None

    @property
    def uB(self):
        if self.solvedB:
            return self._remote.reduce('bWave').reshape(self.fieldDims)
        else:
            return None

    @property
    def dPred(self):
        if self.solvedF:
            return self._remote.reduce('dPred')
        else:
            return None

    @property
    def g(self):
        if self.solvedF and self.solvedB:
            return self._remote.reduceMul('fWave', 'bWave', axis=0).reshape(self.modelDims)
        else:
            return None

    @property
    def dObs(self):
        return getattr(self, '_dobs', None)
    @dObs.setter
    def dObs(self, value):
        self._dobs = commonReducer(value)
        self._remote['dObs'] = self._dobs

    def _computeResidual(self):
        if not self.solvedF:
            raise Exception('Forward problem has not been solved yet!')

        if self.dObs is None:
            raise Exception('No observed data has been defined!')

        if getattr(self, '_residualPrecomputed', None) is None:
            self._residualPrecomputed = False

        if not self._residualPrecomputed:
            self._remote.remoteDifferenceGatherFirst('dPred', 'dObs', 'dResid')
            #self._remote.dview.execute('dResid = commonReducer({key: np.log(dResid[key]).real for key in dResid.keys()}')
            self._residualPrecomputed = True

    @property
    def residual(self):
        if self.solvedF:
            self._computeResidual()
            return self._remote.e0['dResid']
        else:
            return None
    # A day may come when it may be useful to set this, or to set dPred; but it is not this day!
    # @residual.setter
    # def residual(self, value):
    #     self._remote['dResid'] = commonReducer(value)

    @property
    def misfit(self):
        if self.solvedF:
            if getattr(self, '_misfit', None) is None:
                self._computeResidual()
                self._misfit = self._remote.normFromDifference('dResid')
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
    def _remoteFieldDims(self):
        return (self.nsrc, self.nz*self.nx)

    def spawnInterfaces(self):

        self.survey = SurveyHelm(self)
        self.problem = ProblemHelm(self)

        self.survey.pair(self.problem)

        return self.survey, self.problem

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
