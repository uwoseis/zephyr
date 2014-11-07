import numpy as np
import SimPEG
from SimPEG import Utils
import scipy.sparse as sp

DEFAULT_FREESURF_BOUNDS = [False, False, False, False]
DEFAULT_PML_SIZE = 10

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

def initHelmholtzNinePoint (sc):
    '''
    An attempt to reproduce the finite-difference stencil and the
    general behaviour of OMEGA by Pratt et al. The stencil is a 9-point
    second-order version based on work by a number of people in the mid-90s
    including Ivan Stekl. The boundary conditions are based on the PML
    implementation by Steve Roecker in fdfdpml.f.
    '''

    # Set up SimPEG mesh
    hx = [(sc['dx'], sc['nx'])]
    hz = [(sc['dz'], sc['nz'])]
    mesh = SimPEG.Mesh.TensorMesh([hx, hz], '00') 
    dims = (sc['nz']+1, sc['nx']+1)
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

    # Set up physical properties in matrices with padding
    omega   = 2 * np.pi * sc['freq']
    cPad    = np.pad(c, pad_width=1, mode='edge')
    rhoPad  = np.pad(rho, pad_width=1, mode='edge')

    # Wavenumber for 2.5D case
    aky = 2*np.pi*sc.get('ky', 0)

    # Model parameter M
    K = ((omega**2 / cPad**2) - aky**2) / rhoPad

    # Horizontal, vertical and diagonal geometry terms
    dx  = sc['dx']
    dz  = sc['dz']
    dxx = dx**2
    dzz = dz**2
    dxz = dx*dz
    dd  = np.sqrt(dxz)

    # PML decay terms
    # NB: Arrays are padded later, but 'c' in these lines
    #     comes from the original (un-padded) version

    if 'nPML' in sc:
        nPML    = sc['nPML']
    else:
        nPML    = DEFAULT_PML_SIZE

    pmldx   = dx*(nPML - 1)
    pmldz   = dz*(nPML - 1)
    pmlr    = 1e-3
    pmlfx   = 3.0 * np.log(1/pmlr)/(2*pmldx**3)
    pmlfz   = 3.0 * np.log(1/pmlr)/(2*pmldz**3)

    dpmlx   = np.zeros(dims, dtype=np.complex128)
    dpmlz   = np.zeros(dims, dtype=np.complex128)
    isnx    = np.zeros(dims, dtype=np.float64)
    isnz    = np.zeros(dims, dtype=np.float64)

    # Only enable PML if the free surface isn't set

    freeSurf = sc['freeSurf']

    if freeSurf[0]:    
        isnz[-nPML:,:] = -1 # Top

    if freeSurf[1]:
        isnx[:,-nPML:] = -1 # Right Side

    if freeSurf[2]:
        isnz[:nPML,:] = 1 # Bottom

    if freeSurf[3]:
        isnx[:,:nPML] = 1 # Left side

    dpmlx[:,:nPML] = (np.arange(nPML, 0, -1)*dx).reshape((1,nPML))
    dpmlx[:,-nPML:] = (np.arange(1, nPML+1, 1)*dx).reshape((1,nPML))
    dnx     = pmlfx*c*dpmlx**2
    ddnx    = 2*pmlfx*c*dpmlx
    denx    = dnx + 1j*omega
    r1x     = 1j*omega / denx
    r1xsq   = r1x**2
    r2x     = isnx*r1xsq*ddnx/denx

    dpmlz[:nPML,:] = (np.arange(nPML, 0, -1)*dz).reshape((nPML,1))
    dpmlz[-nPML:,:] = (np.arange(1, nPML+1, 1)*dz).reshape((nPML,1))
    dnz     = pmlfz*c*dpmlz**2
    ddnz    = 2*pmlfz*c*dpmlz
    denz    = dnz + 1j*omega
    r1z     = 1j*omega / denz
    r1zsq   = r1z**2
    r2z     = isnz*r1zsq*ddnz/denz

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
        'AD':   (-1) * dims[1] + (-1), 
        'DD':   (-1) * dims[1] + ( 0),
        'CD':   (-1) * dims[1] + (+1),
        'AA':   ( 0) * dims[1] + (-1),
        'BE':   ( 0) * dims[1] + ( 0),
        'CC':   ( 0) * dims[1] + (+1),
        'AF':   (+1) * dims[1] + (-1),
        'FF':   (+1) * dims[1] + ( 0),
        'CF':   (+1) * dims[1] + (+1),
    }


    # Buoyancies
    bMM = 1. / rhoPad[0:-2,0:-2] # bottom left
    bME = 1. / rhoPad[0:-2,1:-1] # bottom centre
    bMP = 1. / rhoPad[0:-2,2:  ] # bottom centre
    bEM = 1. / rhoPad[1:-1,0:-2] # middle left
    bEE = 1. / rhoPad[1:-1,1:-1] # middle centre
    bEP = 1. / rhoPad[1:-1,2:  ] # middle right
    bPM = 1. / rhoPad[2:  ,0:-2] # top    left
    bPE = 1. / rhoPad[2:  ,1:-1] # top    centre
    bPP = 1. / rhoPad[2:  ,2:  ] # top    right

    # Initialize averaged buoyancies on most of the grid
    bMM = (bEE + bMM) / 2 # a2
    bME = (bEE + bME) / 2 # d1
    bMP = (bEE + bMP) / 2 # d2
    bEM = (bEE + bEM) / 2 # a1
    # ... middle
    bEP = (bEE + bEP) / 2 # c1
    bPM = (bEE + bPM) / 2 # f2
    bPE = (bEE + bPE) / 2 # f1
    bPP = (bEE + bPP) / 2 # c2

    # Reset the buoyancies on the outside edges
    bMM[ 0, :] = bEE[ 0, :]
    bMM[ :, 0] = bEE[ :, 0]
    bME[ 0, :] = bEE[ 0, :]
    bMP[ 0, :] = bEE[ 0, :]
    bMP[ :,-1] = bEE[ :,-1]
    bEM[ :, 0] = bEE[ :, 0]
    bEP[ :,-1] = bEE[ :,-1]
    bPM[-1, :] = bEE[-1, :]
    bPM[ :, 0] = bEE[ :, 0]
    bPE[-1, :] = bEE[-1, :]
    bPP[-1, :] = bEE[-1, :]
    bPP[ :,-1] = bEE[ :,-1]

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

    # 9-point fd star
    acoef   = 0.5461
    bcoef   = 0.4539
    ccoef   = 0.6248
    dcoef   = 0.09381
    ecoef   = 0.000001297

    # 5-point fd star
    # acoef = 1.0
    # bcoef = 0.0
    # ecoef = 0.0

    # NB: bPM and bMP here are switched relative to S. Roecker's version
    #     in OMEGA. This is because the labelling herein is always ?ZX.

    diagonals = {
        'AD':   ecoef*kMM
                + bcoef*bMM*((r1zsq+r1xsq)/(4*dxz) - (r2z+r2x)/(4*dd)),
        'DD':   dcoef*kME
                + acoef*bME*(r1zsq/dz - r2z/2)/dz
                + bcoef*(r1zsq-r1xsq)*(bMP+bMM)/(4*dxz),
        'CD':   ecoef*kMP
                + bcoef*bMP*((r1zsq+r1xsq)/(4*dxz) - (r2z+r2x)/(4*dd)),
        'AA':   dcoef*kEM
                + acoef*bEM*(r1xsq/dx - r2x/2)/dx
                + bcoef*(r1xsq-r1zsq)*(bPM+bMM)/(4*dxz),
        'BE':   ccoef*kEE
                + acoef*(r2x*(bEM-bEP)/(2*dx) + r2z*(bME-bPE)/(2*dz) - r1xsq*(bEM+bEP)/dxx - r1zsq*(bME+bPE)/dzz)
                + bcoef*(((r2x+r2z)*(bMM-bPP) + (r2z-r2x)*(bMP-bPM))/(4*dd) - (r1xsq+r1zsq)*(bMM+bPP+bPM+bMP)/(4*dxz)),
        'CC':   dcoef*kEP
                + acoef*bEP*(r1xsq/dx + r2x/2)/dx
                + bcoef*(r1xsq-r1zsq)*(bMP+bPP)/(4*dxz),
        'AF':   ecoef*kPM
                + bcoef*bPM*((r1zsq+r1xsq)/(4*dxz) - (r2z+r2x)/(4*dd)),
        'FF':   dcoef*kPE
                + acoef*bPE*(r1zsq/dz - r2z/2)/dz
                + bcoef*(r1zsq-r1xsq)*(bPM+bPP)/(4*dxz),
        'CF':   ecoef*kPP
                + bcoef*bPP*((r1zsq+r1xsq)/(4*dxz) - (r2z+r2x)/(4*dd)),
    }

    if 'freeSurf' in sc:
        setupBoundary(diagonals, sc['freeSurf'])
    else:
        setupBoundary(diagonals, DEFAULT_FREESURF_BOUNDS)

    diagonals = np.array([diagonals[key].ravel() for key in keys])
    offsets = [offsets[key] for key in keys]

    A = sp.spdiags(diagonals, offsets, mesh.nN, mesh.nN, format='csr')

    return mesh, A