
import unittest
import numpy as np

from zephyr.backend.meta import AttributeMapper

class TestMeta(unittest.TestCase):

    def setUp(self):
        pass

    def test_mapperAsExpected(self):
        
        class TestAttributeMapper(AttributeMapper):
            
            initMap = {
            #   Argument        Required    Rename as ...   Store as type
                'a':            (False,     '_a',          np.complex128),
                'b':            (False,     None,          np.float64),
                'c':            (False,     None,          np.int64),
                'd':            (False,     '_d',          list),
                'e':            (False,     '_e',          tuple),
                'f':            (False,     '_f',          None),
                'g':            (False,     None,          None),
                'h':            (False,     '_dnexist',    np.int64),
                'i':            (True,      '_mustexist',  np.int64),
            }
            
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

        with self.assertRaises(Exception):
            tam = TestAttributeMapper(systemConfig)

        systemConfig.update({'i': 0})

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
