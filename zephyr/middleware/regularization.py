
import SimPEG
from .maps import SquaredSlownessMap

class HelmBaseRegularization(SimPEG.Regularization.BaseRegularization):
    
    mapPair = SquaredSlownessMap
    
    @property
    def W(self):
        """Full regularization weighting matrix W."""
        return self.mesh.aveN2CC.T
