import numpy as np
import SimPEG
from SimPEG import Utils
import scipy.sparse as sp

DEFAULTBOUNDS = [False, False, False, False]

# NOT CONVINCED THIS WORKS
def setupBoundary (diagonals, freeSurf):

    keys = diagonals.keys()
    pickDiag = lambda x: -1. if freeSurf[x] else 1.

    # Left
    for key in keys:
        if key is 'BE':
            diagonals[key][:,0] = pickDiag(3)
        else:
            diagonals[key][:,0] = 0.

    # Right
    for key in keys:
        if key is 'BE':
            diagonals[key][:,-1] = pickDiag(1)
        else:
            diagonals[key][:,-1] = 0.

    # Bottom
    for key in keys:
        if key is 'BE':
            diagonals[key][0,:] = pickDiag(2)
        else:
            diagonals[key][0,:] = 0.

    # Top
    for key in keys:
        if key is 'BE':
            diagonals[key][-1,:] = pickDiag(0)
        else:
            diagonals[key][-1,:] = 0.

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

    # Model parameter M
    K = omega**2 / (c**2 * rho)

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

    # K = omega^2/(c^2 . rho)
    kMM = K[0:-2,0:-2] # bottom left
    kME = K[0:-2,1:-1] # bottom centre
    kMP = K[0:-2,2:  ] # bottom centre
    kEM = K[1:-1,0:-2] # middle left
    kEE = K[1:-1,1:-1] # middle centre
    kEP = K[1:-1,2:  ] # middle right
    kPM = K[2:  ,0:-2] # top    left
    kPE = K[2:  ,1:-1] # top    centre
    kPP = K[2:  ,2:  ] # top    right

    # Initialize outside edges
    cMM = bEE.copy() / dxz # a2
    cME = bEE.copy() / dzz # d1
    cMP = bEE.copy() / dxz # d2
    cEM = bEE.copy() / dxx # a1
    # ... middle
    cEP = bEE.copy() / dxx # c1
    cPM = bEE.copy() / dxz # f2
    cPE = bEE.copy() / dzz # f1
    cPP = bEE.copy() / dxz # c2

    # Reciprocal of the mass in each diagonal on the cell grid
    cMM[1:  ,1:  ]  = (bEE[1:  ,1:  ] + bMM[1:  ,1:  ]) / (2 * dxz) # a2
    cME[1:  , :  ]  = (bEE[1:  , :  ] + bME[1:  , :  ]) / (2 * dzz) # d1
    cMP[1:  , :-1]  = (bEE[1:  , :-1] + bMP[1:  , :-1]) / (2 * dxz) # d2
    cEM[ :  ,1:  ]  = (bEE[ :  ,1:  ] + bEM[ :  ,1:  ]) / (2 * dxx) # a1
    # ... middle
    cEP[ :  , :-1]  = (bEE[ :  , :-1] + bEP[ :  , :-1]) / (2 * dxx) # c1
    cPM[ :-1,1:  ]  = (bEE[ :-1,1:  ] + bPM[ :-1,1:  ]) / (2 * dxz) # f2
    cPE[ :-1, :  ]  = (bEE[ :-1, :  ] + bPE[ :-1, :  ]) / (2 * dzz) # f1
    cPP[ :-1, :-1]  = (bEE[ :-1, :-1] + bPP[ :-1, :-1]) / (2 * dxz) # c2

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
        'AD':   ecoef*kMM + bcoef*cMM,
        'DD':   dcoef*kME + acoef*cME,
        'CD':   ecoef*kMP + bcoef*cMP,
        'AA':   dcoef*kEM + acoef*cEM,
        'BE':   ccoef*kEE - acoef*(cEM+cEP+cME+cPE) - bcoef*(cMM+cPP+cMP+cPM),
        'CC':   dcoef*kEP + acoef*cEP,
        'AF':   ecoef*kPM + bcoef*cPM,
        'FF':   dcoef*kPE + acoef*cPE,
        'CF':   ecoef*kPP + bcoef*cPP,
    }

    # NOT CONVINCED THIS WORKS
    if 'freeSurf' in sc:
        setupBoundary(diagonals, sc['freeSurf'])
    else:
        setupBoundary(diagonals, DEFAULTBOUNDS)

    diagonals = np.array([diagonals[key].ravel() for key in keys])
    offsets = [offsets[key] for key in keys]

    A = sp.spdiags(diagonals, offsets, mesh.nN, mesh.nN, format='csr')

    return mesh, A