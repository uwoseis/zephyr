
import SimPEG

EPS = 1e-10

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

class SquaredSlownessMap(NodalIdentityMap):

    @property
    @staticmethod
    def eps():
        return EPS

    def _transform(self, m):

        m = NodalIdentityMap._transform(self, m)

        return 1. / (m**2 + EPS)

    def inverse(self, D):

        D = 1. / (np.sqrt(D) + EPS)

        return NodalIdentityMap._transform(self, D)

    def deriv(self, m):

        m = NodalIdentityMap._transform(self, m)

        return 1. / (m**2 + EPS)

