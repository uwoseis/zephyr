import numpy as np
import SimPEG
from SimPEG import Utils

def initHelmholtzSimPEG (sc):
    
    hx = np.ones(sc['nx']) * sc['dx']
    hz = np.ones(sc['nz']) * sc['dz']
    mesh = SimPEG.Mesh.TensorMesh([hx, hz], '00')
    omega = 2 * np.pi * sc['freq']
    
    if sc['Q'] is np.inf:
        c = sc['c']
    else:
        c = sc['c'] + (-1j * sc['c'] / (2*sc['Q']))

    c = c.ravel()
    
    L = mesh.nodalLaplacian # (nx+1 * nz+1) by (nx+1 * nz+1)
    mAve = mesh.aveN2CC     #   (nx * nz)   by (nx+1 * nz+1)
    k2 = Utils.sdiag(mAve.T * (omega**2 / c**2))
    A = L + k2

    Ainv = SimPEG.SolverLU(A)
    
    return mesh, A, Ainv
