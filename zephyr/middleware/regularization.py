
import SimPEG
from .maps import NodalIdentityMap

class HelmBaseRegularization(SimPEG.Regularization.BaseRegularization):
    
    mapPair = NodalIdentityMap
    
    @property
    def W(self):
        """Full regularization weighting matrix W."""
        return self.mesh.aveN2CC.T
