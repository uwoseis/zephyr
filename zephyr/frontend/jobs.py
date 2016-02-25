
from zephyr import backend
from zephyr import middleware
from zephyr import frontend

class Job(object):
    '''
    The base class for jobs.
    '''

    Problem = None
    Survey = None
    SystemWrapper = None
    Disc = None

    def __init__(self, projnm, supplementalConfig=None):

        self.projnm = projnm

        print('Setting up composite job "%s":'%(self.__class__.__name__,))
        for item in self.__class__.__mro__[:-1][::-1]:
            print('\t%s'%(item.__name__,))

        proj = self.getProject(projnm)
        systemConfig = proj.systemConfig
        update = {}

        if self.SystemWrapper is not None:
            update['SystemWrapper'] = self.SystemWrapper

        if self.Disc is not None:
            update['Disc'] = self.Disc

        systemConfig.update(update)
        if supplementalConfig is not None:
            systemConfig.update(supplementalConfig)

        # Set up problem and survey objects
        self.problem = self.Problem(systemConfig)
        self.survey = self.Survey(systemConfig)
        self.problem.pair(self.survey)

    def getProject(self, projnm):
        '''
        Get the project
        '''

        raise NotImplementedError

    def run(self):
        '''
        Run the job
        '''

        raise NotImplementedError

    def saveData(self, data):
        '''
        Output the data
        '''

        raise NotImplementedError


class ForwardModelingJob(Job):
    '''
    A task job that selects forward modelling.
    '''

    def run(self):

        messageInfo = {
            'class':    self.__class__.__name__,
            'projnm':   self.projnm,
        }

        print('Running %(class)s(%(projnm)s)...'%messageInfo)

        print('\t- solving system')
        data = self.survey.dpred()

        print('\t- saving data')
        self.saveData(data)

        print('Done!')


class Visco2DJob(Job):
    '''
    A physics job profile that selects 2D viscoacoustic Helmholtz
    '''

    Problem = middleware.Helm2DViscoProblem
    Survey = middleware.Helm2DSurvey


class IsotropicVisco2DJob(Visco2DJob):
    '''
    A physics job profile that selects 2D viscoacoustic Helmholtz
    with isotropy (i.e., MiniZephyr).
    '''

    Disc = backend.MiniZephyrHD


class AnisotropicVisco2DJob(Visco2DJob):
    '''
    A physics job profile that selects 2D viscoacoustic Helmholtz
    with TTI anisotropy (i.e., Eurus).
    '''

    Disc = backend.EurusHD


class FullwvIOJob(Job):
    '''
    An output job profile that saves results to a projnm.utout file
    '''

    def getProject(self, projnm):

        self.fds = middleware.FullwvDatastore(projnm)
        return self.fds

    def saveData(self, data):

        self.fds.utoutWrite(data)

class OmegaJob(IsotropicVisco2DJob, ForwardModelingJob, FullwvIOJob):

    '''
    A 2D viscoacoustic parallel job on the local machine.
    Roughly equivalent to the default behaviour of OMEGA.
    '''
    pass

class AnisoOmegaJob(AnisotropicVisco2DJob, ForwardModelingJob, FullwvIOJob):

    '''
    A 2D viscoacoustic parallel job on the local machine.
    Roughly equivalent to the default behaviour of OMEGA.
    Replaces isotropic solver with TTI anisotropic solver.
    '''
    pass

