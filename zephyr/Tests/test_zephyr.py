import unittest
import numpy as np

IPYPROFILE = 'mpi'
PARNWORKERS = 4

class TestZephyr(unittest.TestCase):

    def setUp(self):
        pass

    # def requireParallel(self):
    #     if not getattr(self, 'parallelActive', False):
    #         import os
    #         import time
    #         os.system('ipcluster start --profile %(profile)s -n %(nworkers)d --daemon'%{'profile': IPYPROFILE, 'nworkers': PARNWORKERS})
    #         time.sleep(10)

    def getBaseConfig(self):
        import numpy as np

        cellSize = 1
        nx = 100
        nz = 100
        nPML = 10

        nsrc = 10
        nrec = 10

        dens = 2700
        Q = np.inf

        vel = 2500
        vanom = -0.05
        anomsize = 10

        rho = np.ones((nz,nx)) * dens
        c = np.ones((nz,nx)) * vel
        c[(nz/2)-(anomsize/2):(nz/2)+(anomsize/2),(nx/2)-(anomsize/2):(nx/2)+(anomsize/2)] += vanom*vel

        freeSurf = [False, False, False, False]

        geom = {
            'srcs': np.array([np.ones(nsrc)*(nPML*cellSize), np.zeros(nsrc), np.linspace((nPML*cellSize), (nz-nPML)*cellSize, nsrc)]).T,
            'recs': np.array([np.ones(nrec)*((nx-nPML)*cellSize), np.zeros(nrec), np.linspace((nPML*cellSize), (nz-nPML)*cellSize, nrec)]).T,
            'mode': 'fixed',
        }

        systemConfig = {
            'dx':       cellSize,   # m
            'dz':       cellSize,   # m
            'c':        c,          # m/s
            'rho':      rho,        # density
            'Q':        Q,          # can be inf
            'nx':       nx,         # count
            'nz':       nz,         # count
            'freeSurf': freeSurf,   # t r b l
            'nPML':     nPML,
            'geom':     geom,
        }

        return systemConfig


    def test_forwardModelling(self):

        sc = self.getBaseConfig()
        sc.update({
            'freq': 2e2,
            'ky':   0,
        })
        geom = sc['geom']

        from zephyr.Survey import HelmSrc, HelmRx
        rxs = [HelmRx(loc, 1.) for loc in geom['recs']]
        sx  = HelmSrc(geom['srcs'][0], 1., rxs)

        from zephyr.Kernel import SeisFDFDKernel
        sp = SeisFDFDKernel(sc)

        u, d = sp.forward(sx, False)
        u.shape = (sc['nz'],sc['nx'])

    def test_parallelForwardModelling(self):

        sc = self.getBaseConfig()
        sc.update({
            'freqs':    [1e2, 2e2, 3e2, 4e2],
            'nky':      1,
            'profile':  IPYPROFILE,
        })

        from zephyr.Dispatcher import SeisFDFDDispatcher

        sp = SeisFDFDDispatcher(sc)
        survey, problem = sp.spawnInterfaces()
        srcs = survey.genSrc()
        sp.srcs = srcs

        sp.forward()
        self.assertTrue(sp.solvedF)

        for node in sp.forwardGraph:
            for job in sp.forwardGraph.node[node].get('jobs', []):
                self.assertTrue('error' not in job.status)

    def test_DoSomething(self):
        #self.requireParallel()
        pass

    # def __del__(self):
    #     if getattr(self, 'parallelActive', False):
    #         import os
    #         os.system('ipcluster stop --profile mpi')

if __name__ == '__main__':
    unittest.main()
