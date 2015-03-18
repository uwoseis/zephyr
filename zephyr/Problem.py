import SimPEG
import numpy
from zephyr.Survey import SurveyHelm


class ProblemHelm(SimPEG.Problem.BaseProblem):

    surveyPair = SurveyHelm

    def __init__(self, dispatcher, **kwargs):

        self.dispatcher = dispatcher
        SimPEG.Problem.BaseProblem.__init__(self, dispatcher.mesh, **kwargs)
        SimPEG.Utils.setKwargs(self, **kwargs)

    def fields(self, c):

        self.dispatcher._rebuildSystem(c)
        txs = self.survey.txList

        u, d = self.dispatcher.forward(txs, block=True, dOnly=False)

        return u
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