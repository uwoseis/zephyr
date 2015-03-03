import numpy as np
from IPython.parallel import Reference, interactive
from SimPEG import Survey, Problem, Mesh, Solver as SimpegSolver
from zephyr.Util import commonReducer
from zephyr.Parallel import RemoteInterface
import networkx

DEFAULT_MPI = True
MPI_BELLWETHERS = ['PMI_SIZE', 'OMPI_UNIVERSE_SIZE']


@interactive
def setupSystem(scu):

    import os
    import zephyr.Kernel as Kernel
    from IPython.parallel.error import UnmetDependency

    global localSystem
    global localLocator

    tag = (scu['ifreq'], scu['iky'])

    # If there is already a system to do this job on this machine, push the duplicate to another
    if tag in localSystem:
        raise UnmetDependency

    subSystemConfig = baseSystemConfig.copy()
    subSystemConfig.update(scu)

    # Set up method output caching
    if 'cacheDir' in baseSystemConfig:
        subSystemConfig['cacheDir'] = os.path.join(baseSystemConfig['cacheDir'], 'cache', '%d-%d'%tag)

    localSystem[tag] = Kernel.SeisFDFDKernel(subSystemConfig, locator=localLocator)

    return tag

# def blockOnTag(fn):
#     def checkForSystem(*args, **kwargs):
#         from IPython.parallel.error import UnmetDependency
#         if not args[0] in localSystem:
#             raise UnmetDependency

#         return fn(*args, **kwargs)

#     return checkForSystem


@interactive
def setupCommon():
    global baseSystemConfig

    localLocator = Kernel.SeisLocator25D(subSystemConfig['geom'])

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

    if not key in dataResultTracker:
        dims = (localLocator.nsrc, localLocator.nrec)
        dataResultTracker[key] = np.zeros(dims, dtype=np.complex128)

    if not key in forwardResultTracker:
        dims = (localLocator.nsrc, localSystem[tag].mesh.nN)
        forwardResultTracker[key] = np.zeros(dims, dtype=np.complex128)

    u, d = localSystem[tag].forward(isrc, dOnly=False, **kwargs)
    forwardResultTracker[key][isrc] += u
    dataResultTracker[key][isrc] += d

@interactive
# @blockOnTag
def forwardFromTagAccumulateAll(tag, isrcs, **kwargs):

    for isrc in isrcs:
        forwardFromTagAccumulate(tag, isrc, **kwargs)

@interactive
# @blockOnTag
def backpropFromTagAccumulate(tag, isrc):

    from IPython.parallel.error import UnmetDependency
    if not tag in localSystem:
        raise UnmetDependency

    key = tag[0]

    if not key in backpropResultTracker:
        dims = (localLocator.nsrc, localSystem[tag].mesh.nN)
        backpropResultTracker[key] = np.zeros(dims, dtype=np.complex128)

    u = localSystem[tag].backprop(isrc, **kwargs)
    backpropResultTracker[key][isrc] += u

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


class SeisFDFDProblem(Problem.BaseProblem):
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

        hx = [self.systemConfig['dx'], self.systemConfig['nx']-1]
        hz = [self.systemConfig['dz'], self.systemConfig['nz']-1]
        mesh = Mesh.TensorMesh([hx, hz], '00')

        # NB: Remember to set up something to do geometry conversion
        #     from origin geometry to local geometry. Functions that
        #     wrap the geometry vectors are probably easiest.

        Problem.BaseProblem.__init__(self, mesh, **kwargs)


        splitkeys = ['freqs', 'nky']

        subConfigSettings = {}
        for key in splitkeys:
            value = self.systemConfig.pop(key, None)
            if value is not None:
                subConfigSettings[key] = value

        self._subConfigSettings = subConfigSettings

        self._remote = RemoteInterface(systemConfig)
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

        self._rebuildSystem()


    def _getHandles(self, systemConfig, subConfigSettings):

        pclient = self._remote.pclient
        dview = self._remote.dview
        lview = self._remote.lview

        subConfigs = self._gen25DSubConfigs(**subConfigSettings)
        nsp = len(subConfigs)

        # Set up dictionary for subproblem objects and push base configuration for the system
        dview['localSystem'] = {}
        self._remote['baseSystemConfig'] = systemConfig # Faster if MPI is available
        dview['dataResultTracker'] = commonReducer()
        dview['forwardResultTracker'] = commonReducer()
        dview['backpropResultTracker'] = commonReducer()

        dview.execute("localLocator = Kernel.SeisLocator25D(baseSystemConfig['geom'])")


        # Create a function to get a subproblem forward modelling function
        dview['forwardFromTag'] = lambda tag, isrc, dOnly=True: localSystem[tag].forward(isrc, dOnly)
        forwardFromTag = Reference('forwardFromTag')

        # Create a function to get a subproblem gradient function
        dview['gradientFromTag'] = lambda tag, isrc, dresid=1.: localSystem[tag].gradient(isrc, dresid)
        gradientFromTag = Reference('gradientFromTag')

        dview['forwardFromTagAccumulate'] = forwardFromTagAccumulate
        dview['forwardFromTagAccumulateAll'] = forwardFromTagAccumulateAll
        dview['clearFromTag'] = clearFromTag

        dview.wait()

        if 'parFac' in systemConfig:
            parFac = systemConfig['parFac']
        else:
            parFac = 1

        while parFac > 0:
            tags = lview.map_sync(setupSystem, subConfigs)
            parFac -= 1

        # Forward model in 2.5D (in parallel) for an arbitrary source location
        # TODO: Write code to handle multiple data residuals for nom>1
        handles = {
            'forward':  lambda isrc, dOnly=True: reduce(np.add, dview.map(forwardFromTag, tags, [isrc]*nsp, [dOnly]*nsp)),
            'forwardSep': lambda isrc, dOnly=True: dview.map_sync(forwardFromTag, tags, [isrc]*nsp, [dOnly]*nsp),
            'gradient': lambda isrc, dresid=1.0: reduce(np.add, dview.map(gradientFromTag, tags, [isrc]*nsp, [dresid]*nsp)),
            'gradSep':  lambda isrc, dresid=1.0: dview.map_sync(gradientFromTag, tags, [isrc]*nsp, [dresid]*nsp),
    #from __future__ import print_function
    #        'clear':    lambda: print('Cleared stored matrix terms for %d systems.'%len(dview.map_sync(clearFromTag, tags))),
        }

        return handles

    def _gen25DSubConfigs(self, freqs, nky, cmin):
        result = []
        weightfac = 1/(2*nky - 1) if nky > 1 else 1# alternatively, 1/dky
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
    def forward(self, isrcs=None, **kwargs):

        dview = self._remote.dview
        dview['dataResultTracker'] = commonReducer()
        dview['forwardResultTracker'] = commonReducer()

        G = self._systemSolve(Reference('forwardFromTagAccumulateAll'), isrcs)

        # self._remote.lview.wait(G.predecessors('End'))

        # d = self._remote.reduce('dataResultTracker')

        # if not kwargs.get('dOnly', True):
        #     uF = self._remote.reduce('forwardResultTracker')

        #     return uF, d

        # return d

        return G

    def backprop(self, **kwargs):

        dview = self._remote.dview
        dview['backpropResultTracker'] = commonReducer()

        G = self._systemSolve(Reference('backpropFromTagAccumulateAll'), isrcs)

        # self._remote.lview.wait(G.predecessors('End'))

        # uB = self._remote.reduce('backpropResultTracker')

        return G


    def _systemSolve(self, fnRef, isrcs, clearRef=Reference('clearFromTag'), **kwargs):

        dview = self._remote.dview
        lview = self._remote.lview

        chunksPerWorker = self.systemConfig.get('chunksPerWorker', 1)

        G = networkx.DiGraph()

        mainNode = 'Beginning'
        G.add_node(mainNode)

        # Parse sources
        nsrc = len(self.systemConfig['geom']['src'])
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

    def _rebuildSystem(self, c = None):
        if c is not None:
            self.systemConfig['c'] = c
            self._rebuildSystem()
            return

        self._subConfigSettings['cmin'] = self.systemConfig['c'].min()
        subConfigs = self._gen25DSubConfigs(**self._subConfigSettings)
        nsp = len(subConfigs)

        #self.curModel = self.systemConfig['c'].ravel()
        self._handles = self._getHandles(self.systemConfig, self._subConfigSettings)

    def fields(self, c):

        self._rebuildSystem(c)

        # F = FieldsSeisFDFD(self.mesh, self.survey)

        # for freq in self.survey.freqs:
        #     A = self._initHelmholtzNinePoint(freq)
        #     q = self.survey.getTransmitters(freq)
        #     Ainv = self.Solver(A, **self.solverOpts)
        #     sol = Ainv * q
        #     F[q, 'u'] = sol

        return F

    def Jvec(self, m, v, u=None):
        pass

    def Jtvec(self, m, v, u=None):
        pass
