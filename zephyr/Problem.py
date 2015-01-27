import numpy as np
import scipy as sp
from IPython.parallel import Client, parallel, Reference, require, depend, interactive
from SimPEG import Survey, Problem, Mesh, np, sp, Solver as SimpegSolver
from Kernel import *
import networkx

@interactive
def setupSystem(scu):
    import os
    import zephyr.Kernel as Kernel

    global localSystem
    global localLocator

    tag = (scu['ifreq'], scu['iky'])

    subSystemConfig = baseSystemConfig.copy()
    subSystemConfig.update(scu)

    # Set up method output caching
    if 'cacheDir' in baseSystemConfig:
        subSystemConfig['cacheDir'] = os.path.join(baseSystemConfig['cacheDir'], 'cache', '%d-%d'%tag)

    localLocator = Kernel.SeisLocator25D(subSystemConfig['geom'])
    localSystem[tag] = Kernel.SeisFDFDKernel(subSystemConfig, locator=localLocator)

    return tag

@interactive
def forwardFromTagAccumulate(tag, isrc):
    from IPython.parallel.error import UnmetDependency
    if not tag in localSystem:
        raise UnmetDependency

    resultTracker((tag[0], isrc), localSystem[tag].forward(isrc, True))

@interactive
def forwardFromTagAccumulateAll(tag, isrcs):
    from IPython.parallel.error import UnmetDependency
    if not tag in localSystem:
        raise UnmetDependency

    for isrc in isrcs:
        forwardFromTagAccumulate(tag, isrc)

@interactive
def hasSystem(tag):
    global localSystem
    return tag in localSystem

@interactive
def hasSystemRank(tag, wid):
    global localSystem
    global rank
    return (tag in localSystem) and (rank == wid)

class commonReducer(dict):

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.addcounter = 0
        self.iaddcounter = 0
        self.interactcounter = 0
        self.callcounter = 0

    def __add__(self, other):
        result = commonReducer(self)
        for key in other.keys():
            if key in result:
                result[key] = self[key] + other[key]
            else:
                result[key] = other[key]

        self.addcounter += 1
        self.interactcounter += 1

        return result

    def __iadd__(self, other):
        for key in other.keys():
            if key in self:
                self[key] += other[key]
            else:
                self[key] = other[key]

        self.iaddcounter += 1
        self.interactcounter += 1

        return self

    def copy(self):

        return commonReducer(self)

    def __call__(self, key, result):
        if key in self:
            self[key] += result
        else:
            self[key] = result

        self.callcounter += 1
        self.interactcounter += 1


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

        hx = [self.systemConfig['dx'], self.systemConfig['nx']]
        hz = [self.systemConfig['dz'], self.systemConfig['nz']]
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

        pclient = Client()

        self.par = {
            'pclient':      pclient,
            'dview':        pclient[:],
            'lview':        pclient.load_balanced_view(),
            'nworkers':     len(pclient.ids),
        }

        dview = self.par['dview']
        dview.clear()

        remoteSetup = '''
                        import numpy as np
                        import scipy as scipy
                        import scipy.sparse
                        import mkl
                        import SimPEG
                        import zephyr.Kernel as Kernel
                      ''' 

        for command in remoteSetup.strip().split('\n'):
            dview.execute(command.strip())

        self._rebuildSystem()

    def _getHandles(self, systemConfig, subConfigSettings):

        pclient = self.par['pclient']
        dview = self.par['dview']

        subConfigs = self._gen25DSubConfigs(**subConfigSettings)
        nsp = len(subConfigs)

        # Set up dictionary for subproblem objects and push base configuration for the system
        #setupCache(systemConfig)
        dview['localSystem'] = {}
        dview['baseSystemConfig'] = systemConfig
        dview['resultTracker'] = commonReducer()
        #localSystem = Reference('localSystem')
        #resultTracker = Reference('resultTracker')

        # Create a function to get a subproblem forward modelling function
        dview['forwardFromTag'] = lambda tag, isrc, dOnly=True: localSystem[tag].forward(isrc, dOnly)
        forwardFromTag = Reference('forwardFromTag')

        # Create a function to get a subproblem gradient function
        dview['gradientFromTag'] = lambda tag, isrc, dresid=1.: localSystem[tag].gradient(isrc, dresid)
        gradientFromTag = Reference('gradientFromTag')

        dview['forwardFromTagAccumulate'] = forwardFromTagAccumulate
        dview['forwardFromTagAccumulateAll'] = forwardFromTagAccumulateAll

        # Set up the subproblem objects with each new configuration
        for wid in pclient.ids:
            pclient[wid]['rank'] = wid
        tags = dview.map_sync(setupSystem, subConfigs)

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
    def forwardAccumulate(self, isrcs=None):

        dview = self.par['dview']
        lview = self.par['lview']

        # Create a function to save forward modelling results to the tracker
        dview.execute("setupFromTag = lambda tag: None")
        #dview['setupFromTag'] = lambda tag: None
        setupFromTag = Reference('setupFromTag')

        forwardFromTagAccumulate = Reference('forwardFromTagAccumulate')

        forwardFromTagAccumulateAll = Reference('forwardFromTagAccumulateAll')

        dview.execute("clearFromTag = lambda tag: localSystem[tag].clear()")
        #dview['clearFromTag'] = lambda tag: localSystem[tag].clear()
        clearFromTag = Reference('clearFromTag')

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

        startJobs = {wid: [] for wid in xrange(len(ids))}
        systemJobs = {}
        endJobs = {wid: [] for wid in xrange(len(ids))}
        tailNodes = []

        for tag in tags:

            startJobsLocal = []
            endJobsLocal = []

            tagNode = 'Head: %d, %d'%tag
            G.add_edge(mainNode, tagNode)

            relIDs = []
            for i in xrange(len(ids)):

                systems = systemsOnWorkers[i]
                rank = ids[i]

                try:
                    jobdeps = {'after': endJobs[i][-1]}
                except IndexError:
                    jobdeps = {}

                if tag in systems:
                    relIDs.append(i)
                    with lview.temp_flags(block=False, **jobdeps):
                        job = lview.apply(depend(hasSystemRank, tag, rank)(setupFromTag), tag)
                        startJobsLocal.append(job)
                        startJobs[i].append(job)
                        label = 'Setup: %d, %d, %d'%(tag[0],tag[1],i)
                        G.add_edge(tagNode, label)

            tagNode = 'Init: %d, %d'%tag
            for i in relIDs:
                label = 'Setup: %d, %d, %d'%(tag[0],tag[1],i)
                G.add_edge(label, tagNode)

            systemJobs[tag] = []
            systemNodes = []

            with lview.temp_flags(block=False, after=startJobsLocal):
		
                # for isrc in isrcslist:
                #     job = lview.apply(depend(hasSystem, tag)(forwardFromTagAccumulate), tag, isrc)
                #     systemJobs[tag].append(job)
                job = lview.apply(forwardFromTagAccumulateAll, tag, isrcslist)
                systemJobs[tag].append(job)
                label = 'Compute: %d, %d'%tag#,isrc)
                systemNodes.append(label)
                G.add_edge(tagNode, label)

            tagNode = 'Wrap: %d, %d'%tag
            for label in systemNodes:
                G.add_edge(label, tagNode)

            relIDs = []
            for i in xrange(len(ids)):
                
                systems = systemsOnWorkers[i]
                rank = ids[i]

                if tag in systems:
                    relIDs.append(i)
                    with lview.temp_flags(block=False, after=systemJobs[tag]):
                        job = lview.apply(depend(hasSystemRank, tag, rank)(clearFromTag), tag)
                        endJobsLocal.append(job)
                        endJobs[i].append(job)
                        label = 'Wrap: %d, %d, %d'%(tag[0],tag[1],i)
                        G.add_edge(tagNode, label)

            tagNode = 'Tail: %d, %d'%tag
            for i in relIDs:
                label = 'Wrap: %d, %d, %d'%(tag[0],tag[1],i)
                G.add_edge(label, tagNode)
            tailNodes.append(tagNode)

        endNode = 'End'
        for node in tailNodes:
            G.add_edge(node, endNode)

        jobs = {
            'startJobs':    startJobs,
            'systemJobs':   systemJobs,
            'endJobs':      endJobs,
        }

        # finaljob dependent on endJobs

        return jobs, G

    def _rebuildSystem(self, c = None):
        if c is not None:
            self.systemConfig['c'] = c
            self._rebuildSystem()
            return


        self._subConfigSettings['cmin'] = self.systemConfig['c'].min()
        subConfigs = self._gen25DSubConfigs(**self._subConfigSettings)
        nsp = len(subConfigs)
        self.par['nproblems'] = nsp

        #self.curModel = self.systemConfig['c'].ravel()
        self._handles = self._getHandles(self.systemConfig, self._subConfigSettings)

    def fields(self, c):

        self._rebuildSystem(c)

        F = FieldsSeisFDFD(self.mesh, self.survey)

        for freq in self.survey.freqs:
            A = self._initHelmholtzNinePoint(freq)
            q = self.survey.getTransmitters(freq)
            Ainv = self.Solver(A, **self.solverOpts)
            sol = Ainv * q
            F[q, 'u'] = sol

        return F

    def Jvec(self, m, v, u=None):
        pass

    def Jtvec(self, m, v, u=None):
        pass
