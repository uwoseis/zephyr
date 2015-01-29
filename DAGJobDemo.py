
import numpy as np
import networkx
from zephyr.Problem import SeisFDFDProblem


### Plotting configuration

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
#get_ipython().magic(u'matplotlib inline')


### System / modelling configuration

cellSize    = 1             # m
freqs       = [2e2]         # Hz
density     = 2700          # units of density
Q           = np.inf        # can be inf
nx          = 164           # count
nz          = 264           # count
freeSurf    = [False, False, False, False] # t r b l
dims        = (nx,nz)       # tuple
nPML        = 32
rho         = np.fliplr(np.ones(dims) * density)
nfreq       = len(freqs)    # number of frequencies
nky         = 48            # number of y-directional plane-wave components
nsp         = nfreq * nky   # total number of 2D subproblems

velocity    = 2500          # m/s
vanom       = 500           # m/s
cPert       = np.zeros(dims)
cPert[(nx/2)-20:(nx/2)+20,(nz/2)-20:(nz/2)+20] = vanom
c           = np.fliplr(np.ones(dims) * velocity)
cFlat       = c
c          += np.fliplr(cPert)
cTrue       = c

srcs        = np.array([np.ones(101)*32, np.zeros(101), np.linspace(32, 232, 101)]).T
recs        = np.array([np.ones(101)*132, np.zeros(101), np.linspace(32, 232, 101)]).T
nsrc        = len(srcs)
nrec        = len(recs)
recmode     = 'fixed'

geom        = {
    'src':  srcs,
    'rec':  recs,
    'mode': 'fixed',
}

cache       = False
cacheDir    = '.'

parFac = 1
chunksPerWorker = 1

# Base configuration for all subproblems
systemConfig = {
    'dx':   cellSize,       # m
    'dz':   cellSize,       # m
    'c':        c.T,        # m/s
    'rho':      rho.T,      # density
    'Q':        Q,          # can be inf
    'nx':       nx,         # count
    'nz':       nz,         # count
    'freeSurf': freeSurf,   # t r b l
    'nPML':     nPML,
    'geom':     geom,
    'cache':    cache,
    'cacheDir': cacheDir,
    'freqs':    freqs,
    'nky':      nky,
    'parFac':   parFac,
    'chunksPerWorker':  chunksPerWorker,
}


sp = SeisFDFDProblem(systemConfig)

jobs, G = sp.forwardAccumulate()

def colourCodeNodes(graph):

    def mapColours(value):
        if value < 0:
            return (0, 0, 0)
        elif value == 0:
            return (0, 0, 1)
        elif value == 1:
            return (0, 1, 0)
        elif value == 3:
            return (1, 0, 0)

    def assessStatus(G, node):

        status = -1
        nodeprops = G.node[node]
        if 'job' in nodeprops:
            job = nodeprops['job']
            status = 1. * job.ready()
            if status > 0:
                status += 1. * (not job.successful())

        return status

    nodeMapper = {
        'Beginning':    (0, 0.5, 0),
        'End':          (0.5, 0, 0),
    }

    colours = []
    sizes = []
    stillGoing = 0
    baseSize = 50
    
    for node in graph.nodes():

        colour = nodeMapper.get(node)

        if colour is None:
            status = assessStatus(graph, node)
            if status < 0:
                sizes.append(baseSize)
            else:
                sizes.append(baseSize*3)
            if status == 0:
                stillGoing += 1
            colour = mapColours(status)
        else:
            sizes.append(baseSize*5)

        colours.append(colour)
    
    return colours, sizes, stillGoing

def trackprogress(G, interval=1.0):

    fig = plt.figure()

    def update():
	fig.clf()
        colours, sizes, stillGoing = colourCodeNodes(G)
        networkx.draw_graphviz(G, node_color=colours, node_size=sizes)
        return stillGoing

    while True:
        try:
            plt.pause(interval)
            stillGoing = update()
            if stillGoing == 0:
                break
        except KeyboardInterrupt:
            print('Exiting loop...')
            break

    plt.pause(interval)
    update()
    plt.show()

trackprogress(G, 3.0)
