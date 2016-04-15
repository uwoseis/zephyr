from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import object

import pickle

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
    Solver = None
    projnm = None

    def __init__(self, projnm, supplementalConfig=None):

        try:
            from pymatsolver import MumpsSolver
        except ImportError:
            print('NB: Can\'t import MumpsSolver; falling back to SuperLU')
        else:
            self.Solver = MumpsSolver

        self.projnm = projnm

        print('Setting up composite job "%s":'%(self.__class__.__name__,))
        for item in self.__class__.__mro__[:-1][::-1]:
            print('\t%s'%(item.__name__,))
        print()

        systemConfig = self.getSystemConfig(projnm)
        update = {}

        if self.SystemWrapper is not None:
            update['SystemWrapper'] = self.SystemWrapper

        if self.Disc is not None:
            update['Disc'] = self.Disc

        if self.Solver is not None:
            update['Solver'] = self.Solver

        systemConfig.update(update)
        if supplementalConfig is not None:
            systemConfig.update(supplementalConfig)

        if not 'projnm' in systemConfig:
            systemConfig['projnm'] = projnm

        # Set up problem and survey objects
        self.systemConfig = systemConfig
        self.problem = self.Problem(systemConfig)
        self.survey = self.Survey(systemConfig)
        self.problem.pair(self.survey)

    def getSystemConfig(self, projnm):
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
        data.shape = (self.survey.nrec, self.survey.nsrc, self.survey.nfreq)

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


class IniInputJob(Job):
    '''
    An input job profile that reads configuration from a projnm.ini
    file and SEG-Y model / data files
    '''

    def getSystemConfig(self, projnm):

        self.ds = middleware.FullwvDatastore(projnm)
        return self.ds.systemConfig


class PythonInputJob(Job):
    '''
    An input job profile that gets configuration from a projnm.py file
    '''

    def getSystemConfig(self, projnm):

        self.ds = middleware.FlatDatastore(projnm)
        return self.ds.systemConfig


class PickleInputJob(Job):
    '''
    An input job profile that gets configuration from a projnm.pickle file
    '''

    def getSystemConfig(self, projnm):

        self.ds = middleware.PickleDatastore(projnm)
        return self.ds.systemConfig


class UtoutOutputJob(Job):
    '''
    An output job profile that saves results to a projnm.utout file
    '''

    def saveData(self, data):

        utow = middleware.UtoutWriter(self.systemConfig)
        utow(data)


class PickleOutputJob(Job):
    '''
    An output job profile that saves results to a projnm.pickle file
    '''

    def saveData(self, data):

        with open(self.projnm, 'wb') as fp:
            pickler = pickle.Pickler(fp)
            pickler.dump(data)


class OmegaIOJob(IniInputJob, UtoutOutputJob):
    '''
    An input/output job profile that emulates Omega
    '''


class OmegaJob(IsotropicVisco2DJob, ForwardModelingJob, OmegaIOJob):

    '''
    A 2D viscoacoustic parallel job on the local machine.
    Roughly equivalent to the default behaviour of OMEGA.
    '''


class PythonUtoutJob(IsotropicVisco2DJob, ForwardModelingJob, PythonInputJob, UtoutOutputJob):
    '''
    A 2D viscoacoustic parallel job on the local machine.
    Constructs systemConfig from a Python file, but outputs to projnm.utout.
    '''


class AnisoOmegaJob(AnisotropicVisco2DJob, ForwardModelingJob, OmegaIOJob):

    '''
    A 2D viscoacoustic parallel job on the local machine.
    Roughly equivalent to the default behaviour of OMEGA.
    Replaces isotropic solver with TTI anisotropic solver.
    '''


class AnisoPythonUtoutJob(AnisotropicVisco2DJob, ForwardModelingJob, PythonInputJob, UtoutOutputJob):
    '''
    A 2D viscoacoustic parallel job on the local machine.
    Constructs systemConfig from a Python file, but outputs to projnm.utout.
    '''
