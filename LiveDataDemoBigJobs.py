
import numpy as np
import networkx
from zephyr.Problem import SeisFDFDProblem

# Plotting configuration

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import matplotlib
matplotlib.rcParams.update({'font.size': 20})

# System / modelling configuration

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
}


sp = SeisFDFDProblem(systemConfig)
jobs = sp.forwardAccumulate()


def trackprogress(sp, jobs, interval=1.0):

    systemJobs = jobs['systemJobs']
    jobkeys = systemJobs.keys()
    jobkeys.sort()

    fig = plt.figure()
    ax1 = fig.add_axes([0.1,0.10,0.15,0.85], xlabel='Subproblem', ylabel='Source')
    ax1.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))

    ax2 = fig.add_axes([0.25,0.10,0.75,0.85], xlabel='Receiver')
    im1 = ax2.imshow(np.zeros((nsrc, nrec)), vmin=-50*nky, vmax=50*nky, cmap=cm.bwr)
    im2 = ax1.imshow(np.zeros((nsrc, nsp)), vmin=0, vmax=2, interpolation='nearest', aspect='auto')

    plt.show()

    def update():
        #try:
        #    res = reduce(np.add, sp.par['dview']['resultTracker'])
        #except:
        #    res = {}

        #keys = [(freqs[0], i) for i in range(nrec)]
        #resarr = np.array([res[key] if key in res.keys() else np.zeros(nrec) for key in keys])
        
        status = np.zeros((len(jobkeys),nsrc))
        for i, key in enumerate(jobkeys):
            status[i,:] = 1. * systemJobs[key][0].ready()#np.array([systemJobs[key][j].ready() for j in xrange(1)])
            if systemJobs[key][0].ready():#for j in np.argwhere(status[i,:]):
                status[i,:] += not systemJobs[key][0].successful()
    
        #im1.set_data(resarr.real)
        im2.set_data(status.T)
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        
    while True:
        try:
            plt.pause(interval)
            update()
        except KeyboardInterrupt:
            print('Exiting loop...')
            break
        finally:
            if not reduce(np.add, sp.par['dview']['resultTracker.interactcounter']) < (nsp * nsrc):
                break


trackprogress(sp, jobs)

