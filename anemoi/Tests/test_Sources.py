import unittest
import numpy as np
from anemoi import SimpleSource, StackedSimpleSource, SparseKaiserSource, KaiserSource

class TestSources(unittest.TestCase):
    
    @staticmethod
    def elementNorm(arr):
        return np.sqrt((arr.conj()*arr).sum()) / arr.size

    def setUp(self):
        pass

    def test_cleanExecution(self):

        systemConfig = {
            'nx':       100,
            'nz':       100,
        }

        loc = np.array([[50.,50.],[25.,25.], [80.,80.], [25.,80.]])
        
        ss  = SimpleSource(systemConfig)
        sss = SimpleSource(systemConfig)
        sks = SparseKaiserSource(systemConfig)
        ks  = KaiserSource(systemConfig)

        qss     = ss(loc)
        qsss    = sss(loc)
        qsks    = sks(loc)
        qks     = ks(loc)

    def test_KaiserSource(self):

        systemConfig = {
            'nx':       100,
            'nz':       100,
        }

        loc = np.array([[50.,50.],[25.,25.], [80.,80.], [25.,80.]])

        sks = SparseKaiserSource(systemConfig)
        ks  = KaiserSource(systemConfig)

        qsks    = sks(loc)
        qks     = ks(loc)

        self.assertTrue(self.elementNorm(qsks.toarray() - qks) == 0.)

    def test_KaiserSourceSimpleCase(self):

        systemConfig = {
            'nx':       100,
            'nz':       100,
            'dx':       1.,  # NB: This will not generally be true,
            'dz':       1.,  # but the scaleTerm in KS is 1/(dx*dz).
        }

        loc = np.array([[50.,50.],[25.,25.], [80.,80.], [25.,80.]])

        ss  = SimpleSource(systemConfig)
        ks  = KaiserSource(systemConfig)

        qss = ss(loc)
        qks = ks(loc)

        self.assertTrue(self.elementNorm(qks - qss) < 1e-10)

