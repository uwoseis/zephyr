# Parallel cluster setup
from IPython.parallel import Client, parallel, Reference, require, depend, interactive
pclient = Client()
dview = pclient[:]
lview = pclient.load_balanced_view()
dview.clear()

# Imports synced to worker nodes
with dview.sync_imports():
    import numpy as np
    import scipy
    import scipy.sparse
    import mkl
    import SimPEG

@dview.remote
@require('mkl')
def threadAdjust(nThreads=1):
    mkl.set_num_threads(nThreads)

# threadAdjust(1)

@interactive
def makeObj(refToDict, key, classRef, *args, **kwargs):
    refToDict[key] = classRef(*args, **kwargs)

with dview.sync_imports():
    #from zephyr.Kernel import SeisFDFDKernel, SeisLocator25D
    import zephyr.Kernel as Kernel

@interactive
def gen25DSubConfigs(freqs, nky, cmin):
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

@interactive
def setupCache(systemConfig):

    global localSystem
    global baseSystemConfig

    localSystem = {}
    baseSystemConfig = systemConfig

@dview.parallel(block=True)
@interactive
def setupSystem(systemConfigUpdates):
    import zephyr.Kernel as Kernel

    global localSystem
    global localLocator

    tags = []

    for scu in systemConfigUpdates:
        subSystemConfig = baseSystemConfig.copy()
        subSystemConfig.update(scu)
        localLocator = Kernel.SeisLocator25D(subSystemConfig['geom'])
        tag = '%f-%f'%(scu['freq'], scu['ky'])
        localSystem[tag] = Kernel.SeisFDFDKernel(subSystemConfig, locator=localLocator)
        tags.append(tag)
    return tags

@interactive
def getForward(systemConfig, subConfigSettings):

    subConfigs = gen25DSubConfigs(**subConfigSettings)
    nsp = len(subConfigs)

    # Set up dictionary for subproblem objects and push base configuration for the system
    #setupCache(systemConfig)
    dview['localSystem'] = {}
    dview['baseSystemConfig'] = systemConfig

    # Create a function to get a subproblem forward modelling function
    dview['forwardFromTag'] = lambda tag, isrc: localSystem[tag].forward(isrc)
    forwardFromTag = Reference('forwardFromTag')

    # Set up the subproblem objects with each new configuration
    tags = setupSystem(subConfigs)#dview.map_sync(setupSystem, subConfigs)

    # Forward model in 2.5D (in parallel) for an arbitrary source location
    forward25D = lambda isrc: reduce(np.add, dview.map(forwardFromTag, tags, [isrc]*nsp))

    return forward25D
