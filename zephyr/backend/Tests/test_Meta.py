
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
                'a':            (False,     'Ta',          np.complex128),
                'b':            (False,     None,          np.float64),
                'c':            (False,     None,          np.int64),
                'd':            (False,     'Td',          list),
                'e':            (False,     'Te',          tuple),
                'f':            (False,     'Tf',          None),
                'g':            (False,     None,          None),
                'h':            (False,     'Tdnexist',    np.int64),
                'i':            (True,      'Tmustexist',  np.int64),
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

        self.assertEqual(tam.Ta, 2500.)
        self.assertIsInstance(tam.Ta, np.complex128)

        self.assertEqual(tam.b, 5)
        self.assertIsInstance(tam.b, np.float64)

        self.assertEqual(tam.c, 100)
        self.assertIsInstance(tam.c, np.int64)

        self.assertEqual(tam.Td[2], systemConfig['d'][2])
        self.assertIsInstance(tam.Td, list)

        self.assertEqual(tam.Te[1], systemConfig['e'][1])
        self.assertIsInstance(tam.Te, tuple)

        self.assertIs(tam.Tf, systemConfig['f'])

        self.assertIs(tam.g, systemConfig['g'])

        self.assertFalse(hasattr(tam, 'h'))
        self.assertFalse(hasattr(tam, 'Tdnexist'))


if __name__ == '__main__':
    unittest.main()
