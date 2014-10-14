import numpy as np
import SimPEG
from SimPEG import Utils
import scipy.sparse as sp

# NOT CONVINCED THIS WORKS
def setupFreeSurface (diagonals, freesurf):
    keys = diagonals.keys()

    if freesurf[0]:
        for key in keys:
            if key is 'BE':
                diagonals[key][-1,:] = -1.
            else:
                diagonals[key][-1,:] = 0.

    if freesurf[1]:
        for key in keys:
            if key is 'BE':
                diagonals[key][:,-1] = -1.
            else:
                diagonals[key][:,-1] = 0.

    if freesurf[2]:
        for key in keys:
            if key is 'BE':
                diagonals[key][0,:] = -1.
            else:
                diagonals[key][0,:] = 0.

    if freesurf[3]:
        for key in keys:
            if key is 'BE':
                diagonals[key][:,0] = -1.
            else:
                diagonals[key][:,0] = 0.

def initHelmholtzNinePointCE (sc):

    # Set up SimPEG mesh
    hx = np.ones(sc['nx']) * sc['dx']
    hz = np.ones(sc['nz']) * sc['dz']
    mesh = SimPEG.Mesh.TensorMesh([hx, hz], '00')
    dims = mesh.vnN
    mAve = mesh.aveN2CC

    # Generate a complex velocity vector if Q is not infinite
    if sc['Q'] is np.inf:
        c = sc['c']
    else:
        c = sc['c'] + (-1j * sc['c'] / (2*sc['Q']))

    c = (mAve.T * c.ravel()).reshape(dims)

    # Read density model from initialization dictionary or generate
    # one using Gardner's relation
    if 'rho' in sc:
        rho = sc['rho']
    else:
        # Gardner's relation for P-velocity in m/s and density in kg/m^3
        rho = 310 * sc['c']**0.25

    rho = (mAve.T * rho.ravel()).reshape(dims)

    # fast --> slow is x --> y --> z as Fortran

    # Set up physical properties in matrices
    omega = 2 * np.pi * sc['freq']
    c = np.pad(c, pad_width=1, mode='edge')
    rho = np.pad(rho, pad_width=1, mode='edge')
    K = omega**2 / c**2

    # Visual key for finite-difference terms
    # (per Pratt and Worthington, 1990)
    #
    #   This         Original
    # AF FF CF  vs.  AD DD CD
    # AA BE CC  vs.  AA BE CC
    # AD DD CD  vs.  AF FF CF

    # Set of keys to index the dictionaries
    keys = ['AD', 'DD', 'CD', 'AA', 'BE', 'CC', 'AF', 'FF', 'CF']

    # Diagonal offsets for the sparse matrix formation
    offsets = {
        'AD':   (-1) * dims[0] + (-1), 
        'DD':   (-1) * dims[0] + ( 0),
        'CD':   (-1) * dims[0] + (+1),
        'AA':   ( 0) * dims[0] + (-1),
        'BE':   ( 0) * dims[0] + ( 0),
        'CC':   ( 0) * dims[0] + (+1),
        'AF':   (+1) * dims[0] + (-1),
        'FF':   (+1) * dims[0] + ( 0),
        'CF':   (+1) * dims[0] + (+1),
    }

    # Horizontal, vertical and diagonal geometry terms
    dx  = sc['dx']
    dz  = sc['dz']
    dxx = dx**2
    dzz = dz**2
    dxz = 2*dx*dz

    # Buoyancies
    bMM = 1. / rho[0:-2,0:-2] # bottom left
    bME = 1. / rho[0:-2,1:-1] # bottom centre
    bMP = 1. / rho[0:-2,2:  ] # bottom centre
    bEM = 1. / rho[1:-1,0:-2] # middle left
    bEE = 1. / rho[1:-1,1:-1] # middle centre
    bEP = 1. / rho[1:-1,2:  ] # middle right
    bPM = 1. / rho[2:  ,0:-2] # top    left
    bPE = 1. / rho[2:  ,1:-1] # top    centre
    bPP = 1. / rho[2:  ,2:  ] # top    right

    # k^2
    kMM = K[0:-2,0:-2] # bottom left
    kME = K[0:-2,1:-1] # bottom centre
    kMP = K[0:-2,2:  ] # bottom centre
    kEM = K[1:-1,0:-2] # middle left
    kEE = K[1:-1,1:-1] # middle centre
    kEP = K[1:-1,2:  ] # middle right
    kPM = K[2:  ,0:-2] # top    left
    kPE = K[2:  ,1:-1] # top    centre
    kPP = K[2:  ,2:  ] # top    right

    # Reciprocal of the mass in each diagonal on the cell grid
    a1  = (bEE + bEM) / (2 * dxx)
    c1  = (bEE + bEP) / (2 * dxx)
    d1  = (bEE + bME) / (2 * dzz)
    f1  = (bEE + bPE) / (2 * dzz)
    a2  = (bEE + bMM) / (2 * dxz)
    c2  = (bEE + bPP) / (2 * dxz)
    d2  = (bEE + bMP) / (2 * dxz)
    f2  = (bEE + bPM) / (2 * dxz)

    # 9-point fd star
    acoef = 0.5461
    bcoef = 0.4539
    ccoef = 0.6248
    dcoef = 0.09381
    ecoef = 0.000001297

    # 5-point fd star
    # acoef = 1.0
    # bcoef = 0.0
    # ecoef = 0.0

    diagonals = {
        'AD':   ecoef*kMM + bcoef*a2,
        'DD':   dcoef*kME + acoef*d1,
        'CD':   ecoef*kMP + bcoef*d2,
        'AA':   dcoef*kEM + acoef*a1,
        'BE':   ccoef*kEE - acoef*(a1+c1+d1+f1) - bcoef*(a2+c2+d2+f2),
        'CC':   dcoef*kEP + acoef*c1,
        'AF':   ecoef*kPM + bcoef*f2,
        'FF':   dcoef*kPE + acoef*f1,
        'CF':   ecoef*kPP + bcoef*c2,
    }

    # NOT CONVINCED THIS WORKS
    if 'freeSurf' in sc:
        setupFreeSurface(diagonals, sc['freeSurf'])

    diagonals = np.array([diagonals[key].ravel() for key in keys])
    offsets = [offsets[key] for key in keys]

    A = sp.spdiags(diagonals, offsets, mesh.nN, mesh.nN, format='csr')
    Ainv = SimPEG.SolverLU(A)

    return mesh, A, Ainv