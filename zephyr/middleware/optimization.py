from __future__ import unicode_literals, print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import SimPEG
import scipy.optimize

class Minimize(SimPEG.Optimize.Minimize):
    pass

