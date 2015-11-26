
import SimPEG

class NodalIdentityMap(SimPEG.Maps.IdentityMap):
    
    @property
    def nP(self):
        
        if self.mesh is None:
            return '*'
        return self.mesh.nC
    
    @property
    def shape(self):
        
        if self.mesh is None:
            return ('*', self.mesh.nN)
        return (self.mesh.nC, self.mesh.nN)
    
    def _transform(self, m):
        
        return self.mesh.aveN2CC * m
    
    def inverse(self, D):
        
        return self.mesh.aveN2CC.T * D
    
    def deriv(self, m):
        
        return self.mesh.aveN2CC
