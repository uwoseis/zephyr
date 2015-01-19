import numpy as np
import scipy as sp
from IPython.parallel import Client, parallel, Reference, require, depend, interactive
from SimPEG import Survey, Problem, Mesh, np, sp, Solver as SimpegSolver
from Kernel import *

def setupSystem(scu):
    import os
    import zephyr.Kernel as Kernel

    global localSystem
    global localLocator

    tags = []

    tag = (scu['freq'], scu['ky'])

    subSystemConfig = baseSystemConfig.copy()
    subSystemConfig.update(scu)

    # Set up method output caching
    if 'cacheDir' in baseSystemConfig:
        subSystemConfig['cacheDir'] = os.path.join(baseSystemConfig['cacheDir'], 'cache', '%f-%f'%tag)

    localLocator = Kernel.SeisLocator25D(subSystemConfig['geom'])
    localSystem[tag] = Kernel.SeisFDFDKernel(subSystemConfig, locator=localLocator)

    return tag

class commonReducer(dict):

    def __add__(self, other):
        result = commonReducer(self)
        for key in other.keys():
            if key in result:
                result[key] = self[key] + other[key]
            else:
                result[key] = other[key]

        return result

    def __iadd__(self, other):
        for key in other.keys():
            if key in self:
                self[key] += other[key]
            else:
                self[key] = other[key]

        return self

    def copy(self):

        return commonReducer(self)

    def __call__(self, key, result):
        if key in self:
            self[key] += result
        else:
            self[key] = result


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

    def _getHandles(self, dview, systemConfig, subConfigSettings):

        subConfigs = self._gen25DSubConfigs(**subConfigSettings)
        nsp = len(subConfigs)

        # Set up dictionary for subproblem objects and push base configuration for the system
        #setupCache(systemConfig)
        dview['localSystem'] = {}
        dview['baseSystemConfig'] = systemConfig
        dview['resultTracker'] = commonReducer()
        #localSystem = Reference('localSystem')
        #resultTracker = Reference('resultTracker')

        # Create a function to save forward modelling results to the tracker
        dview.execute("forwardFromTagAccumulate = lambda tag, isrc: resultTracker('%r %03d'%(tag[0], isrc), localSystem[tag].forward(isrc, True))")
        #dview['forwardFromTagAccumulate'] = lambda tag, isrc: resultTracker('%r %r'%(tag[0], isrc), localSystem[tag].forward(isrc, True))
        forwardFromTagAccumulate = Reference('forwardFromTagAccumulate')

        # Create a function to get a subproblem forward modelling function
        dview['forwardFromTag'] = lambda tag, isrc, dOnly=True: localSystem[tag].forward(isrc, dOnly)
        forwardFromTag = Reference('forwardFromTag')

        # Create a function to get a subproblem gradient function
        dview['gradientFromTag'] = lambda tag, isrc, dresid=1.: localSystem[tag].gradient(isrc, dresid)
        gradientFromTag = Reference('gradientFromTag')

        dview['clearFromTag'] = lambda tag: localSystem[tag].clear()
        clearFromTag = Reference('clearFromTag')

        # Set up the subproblem objects with each new configuration
        tags = dview.map_sync(interactive(setupSystem), subConfigs)

        # Forward model in 2.5D (in parallel) for an arbitrary source location
        # TODO: Write code to handle multiple data residuals for nom>1
        handles = {
            'forward':  lambda isrc, dOnly=True: reduce(np.add, dview.map(forwardFromTag, tags, [isrc]*nsp, [dOnly]*nsp)),
            'forwardSep': lambda isrc, dOnly=True: dview.map_sync(forwardFromTag, tags, [isrc]*nsp, [dOnly]*nsp),
            'gradient': lambda isrc, dresid=1.0: reduce(np.add, dview.map(gradientFromTag, tags, [isrc]*nsp, [dresid]*nsp)),
            'gradSep':  lambda isrc, dresid=1.0: dview.map_sync(gradientFromTag, tags, [isrc]*nsp, [dresid]*nsp),
            'forwardAccumulate': lambda isrc: dview.map(forwardFromTagAccumulate, tags, [isrc]*nsp),
    #from __future__ import print_function
    #        'clear':    lambda: print('Cleared stored matrix terms for %d systems.'%len(dview.map_sync(clearFromTag, tags))),
        }

        return handles

    def _gen25DSubConfigs(self, freqs, nky, cmin):
        result = []
        weightfac = 1/(2*nky - 1) # alternatively, 1/dky
        for freq in freqs:
            k_c = freq / cmin
            dky = k_c / (nky - 1)
            for ky in np.linspace(0, k_c, nky):
                result.append({
                    'freq':     freq,
                    'ky':       ky,
                    'kyweight': 2*weightfac if ky != 0 else weightfac,
                })
        return result

    # Fields

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
        self._handles = self._getHandles(self.par['dview'], self.systemConfig, self._subConfigSettings)

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
