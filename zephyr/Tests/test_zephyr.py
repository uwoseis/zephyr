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

    def test_parallelGradient(self):

        sc = self.getBaseConfig()
        sc.update({
            'freqs':    [1e2, 2e2, 3e2, 4e2],
            'nky':      1,
            'profile':  IPYPROFILE,
        })

        cTrue = sc['c']
        cFlat = cTrue[0,0] * np.ones_like(cTrue)

        from zephyr.Problem import SeisFDFD25DParallelProblem
        from zephyr.Survey import SeisFDFD25DSurvey

        problem = SeisFDFD25DParallelProblem(sc)
        survey = SeisFDFD25DSurvey(sc['geom'])
        survey.pair(problem)

        problem.forward()

        self.assertTrue(len(problem.dPred.keys()) == len(sc['freqs']))

        problem.dObs = problem.dPred
        problem.rebuildSystem(cFlat)

        problem.forward()
        self.assertTrue(problem.solvedF)

        for node in problem.forwardGraph:
            for job in problem.forwardGraph.node[node].get('jobs', []):
                self.assertTrue('error' not in job.status)

        problem.backprop()
        self.assertTrue(problem.solvedB)

        for node in problem.backpropGraph:
            for job in problem.backpropGraph.node[node].get('jobs', []):
                self.assertTrue('error' not in job.status)

        normit = lambda x: np.sqrt(reduce(np.add, [np.sum(x[key]**2) for key in x])).real

        gradient = problem.g
        gradnorm = normit(gradient)
        lastgradnorm = gradnorm

        misfit = problem.misfit
        mfnorm = normit(misfit)
        lastmfnorm = mfnorm

        self.assertTrue(len(problem.g.keys()) == len(sc['freqs']))
        self.assertTrue(len(problem.misfit.keys()) == len(sc['freqs']))

        mfformat = '%d\t%g\t%g'
        print('#\tgradnorm\tmisfit')

        for i in xrange(10):
            wrongness = 0.5**i
            c = wrongness*cFlat + (1-wrongness)*cTrue
            problem.rebuildSystem(c)

            problem.forward()
            problem.backprop()

            gradient = problem.g
            gradnorm = normit(gradient)
            misfit = problem.misfit
            mfnorm = normit(misfit)

            if i > 0:
                self.assertTrue(gradnorm <= lastgradnorm)
                self.assertTrue(mfnorm <= lastmfnorm)
                misfit = normit(problem.misfit)

            lastgradnorm = gradnorm
            lastmfnorm = mfnorm

            print(mfformat%(i, gradnorm, mfnorm))

    # def test_parallelGradient(self):

    # def test_DoSomething(self):
    #     #self.requireParallel()
    #     pass

if __name__ == '__main__':
    unittest.main()
