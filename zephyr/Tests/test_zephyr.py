import unittest
import numpy as np

IPYPROFILE = 'mpi'
PARNWORKERS = 4

class TestZephyr(unittest.TestCase):

    def setUp(self):
        pass

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
            'src': np.array([np.ones(nsrc)*(nPML*cellSize), np.zeros(nsrc), np.linspace((nPML*cellSize), (nz-nPML)*cellSize, nsrc)]).T,
            'rec': np.array([np.ones(nrec)*((nx-nPML)*cellSize), np.zeros(nrec), np.linspace((nPML*cellSize), (nz-nPML)*cellSize, nrec)]).T,
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

        from zephyr.Problem import SeisFDFD25DProblem
        problem = SeisFDFD25DProblem(sc)
        survey = problem.survey

        u, d = problem.forward()

    def test_parallelForwardModelling(self):

        sc = self.getBaseConfig()
        sc.update({
            'freqs':    [1e2, 2e2, 3e2, 4e2],
            'nky':      1,
            'profile':  IPYPROFILE,
        })

        from zephyr.Problem import SeisFDFD25DParallelProblem
        from zephyr.Survey import SeisFDFD25DSurvey

        problem = SeisFDFD25DParallelProblem(sc)
        survey = SeisFDFD25DSurvey(sc['geom'])
        survey.pair(problem)

        problem.forward()
        self.assertTrue(problem.solvedF)

        for node in problem.forwardGraph:
            for job in problem.forwardGraph.node[node].get('jobs', []):
                self.assertTrue('error' not in job.status)

    def test_DoSomething(self):
        #self.requireParallel()
        pass

if __name__ == '__main__':
    unittest.main()
