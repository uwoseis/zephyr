from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import SimPEG
from .maps import SquaredSlownessMap

class HelmBaseRegularization(SimPEG.Regularization.BaseRegularization):
    
    mapPair = SquaredSlownessMap
    
    @property
    def W(self):
        """Full regularization weighting matrix W."""
        return self.mesh.aveN2CC.T
