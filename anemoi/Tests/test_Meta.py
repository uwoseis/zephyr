
import unittest
import numpy as np

from anemoi.meta import AttributeMapper

class TestMeta(unittest.TestCase):

    def setUp(self):
        pass

    def test_mapperAsExpected(self):
        
        class TestAttributeMapper(AttributeMapper):
            
            initMap = {
            #   Argument        Rename as ...   Store as type
                'a':            ('_a',          np.complex128),
                'b':            (None,          np.float64),
                'c':            (None,          np.int64),
                'd':            ('_d',          list),
                'e':            ('_e',          tuple),
                'f':            ('_f',          None),
                'g':            (None,          None),
                'h':            ('_dnexist',    np.int64),
            }
            
            def __init__(self, systemConfig):
                
                super(TestAttributeMapper, self).__init__(systemConfig)

        class TestType(object):
            
            pass
        
        systemConfig = {
            'a':    2500.,
            'b':    5 + 2j,
            'c':    100.1,
            'd':    (1, 2, 3),
            'e':    [4, 5, 6],
            'f':    TestType(),
            'g':    1 + 2j,
        }
        
        tam = TestAttributeMapper(systemConfig)
        
        self.assertEqual(tam._a, 2500.)
        self.assertIsInstance(tam._a, np.complex128)
        
        self.assertEqual(tam.b, 5)
        self.assertIsInstance(tam.b, np.float64)
        
        self.assertEqual(tam.c, 100)
        self.assertIsInstance(tam.c, np.int64)
        
        self.assertEqual(tam._d[2], systemConfig['d'][2])
        self.assertIsInstance(tam._d, list)
        
        self.assertEqual(tam._e[1], systemConfig['e'][1])
        self.assertIsInstance(tam._e, tuple)
        
        self.assertIs(tam._f, systemConfig['f'])
        
        self.assertIs(tam.g, systemConfig['g'])
        
        self.assertFalse(hasattr(tam, 'h'))
        self.assertFalse(hasattr(tam, '_dnexist'))


if __name__ == '__main__':
    unittest.main()
