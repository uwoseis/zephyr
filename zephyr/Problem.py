import SimPEG
import numpy
from zephyr.Survey import SurveyHelm


class ProblemHelm(SimPEG.Problem.BaseProblem):

    surveyPair = SurveyHelm

    def __init__(self, dispatcher, **kwargs):

        self.dispatcher = dispatcher
        SimPEG.Problem.BaseProblem.__init__(self, dispatcher.mesh, **kwargs)
        SimPEG.Utils.setKwargs(self, **kwargs)

    def fields(self, c=None):

        if c is not None:
            self.dispatcher.rebuildSystem(c)

        self.dispatcher.forward()

        return self.dispatcher.uF

    # def Jvec(self, m, v, u=None):
    #     pass

    # def Jtvec(self, m, v, u=None):
    #     pass