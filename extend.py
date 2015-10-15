
# ------------------------------------------------------------------------
# Imports
try:
    import fullwv
except ImportError:
    print('Cannot import \'fullwv\'; callback functions will not work!')

import time

# ------------------------------------------------------------------------
# Module variables
starttime = 0

# ------------------------------------------------------------------------
# Callbacks

def CB_the_start():
    global starttime
    fullwv.dbprint('At the start of the program!')
    starttime = time.time()

    fullwv.pythonSolver[:] = True

def CB_the_end():
    fullwv.dbprint('At the end of the program!')
    timediff = time.time() - starttime
    print('\nTime Elapsed: %8.3f\n'% (timediff,))

# ------------------------------------------------------------------------
# Python discretization from Zephyr

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

class Eurus(object):

    c           =   None
    rho         =   None
    nPML        =   None
    freq        =   None
    ky          =   None
    dx          =   None
    dx          =   None
    nx          =   None
    nz          =   None
    freeSurf    =   None

    def __init__(self, systemConfig):

        initMap = {
        #   Argument        Rename to Property
            'c':            None,
            'rho':          None,
            'nPML':         None,
            'freq':         None,
            'ky':           None,
            'dx':           None,
            'dz':           None,
            'nx':           None,
            'nz':           None,
            'freeSurf':     None,
            'theta':        '_theta',
            'eps':          '_eps',
            'delta':        '_delta',
        }

        for key in initMap.keys():
            if key in systemConfig:
                if initMap[key] is None:
                    setattr(self, key, systemConfig[key])
                else:
                    setattr(self, initMap[key], systemConfig[key])

    def _initHelmholtzNinePoint(self):
        """
        An attempt to reproduce the finite-difference stencil and the
        general behaviour of OMEGA by Pratt et al. The stencil is a 9-point
        second-order version based on work by a number of people in the mid-90s
        including Ivan Stekl. The boundary conditions are based on the PML
        implementation by Steve Roecker in fdfdpml.f.

        SMH 2015: I have modified this code to instead follow the 9-point
        anisotropic stencil suggested by Operto et al. (2009)

        """

        nx = self.nx
        nz = self.nz
        dims = (nz, nx)
        nrows = nx*nz

        c = self.c
        rho = self.rho

        # fast --> slow is x --> y --> z as Fortran

        # Set up physical properties in matrices with padding
        omega   = 2*np.pi * self.freq
        cPad    = np.pad(c, pad_width=1, mode='edge')
        rhoPad  = np.pad(rho, pad_width=1, mode='edge')

        # Horizontal, vertical and diagonal geometry terms
        dx  = self.dx
        dz  = self.dx
        dxx = dx**2
        dzz = dz**2
        dxz = (dxx+dzz)/2
        dd  = np.sqrt(dxz)
        iom = 1j * omega

        # PML decay terms
        # NB: Arrays are padded later, but 'c' in these lines
        #     comes from the original (un-padded) version

        nPML    = self.nPML

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

        freeSurf = self.freeSurf

        if not freeSurf[2]:
            isnz[-nPML:,:] = -1 # Top

        if not freeSurf[1]:
            isnx[:,-nPML:] = -1 # Right Side

        if not freeSurf[0]:
            isnz[:nPML,:] = 1 # Bottom

        if not freeSurf[3]:
            isnx[:,:nPML] = 1 # Left side

        dpmlx[:,:nPML] = (np.arange(nPML, 0, -1)*dx).reshape((1,nPML))
        dpmlx[:,-nPML:] = (np.arange(1, nPML+1, 1)*dx).reshape((1,nPML))
        dnx     = pmlfx*c*dpmlx**2
        ddnx    = 2*pmlfx*c*dpmlx
        denx    = dnx + iom
        r1x     = iom / denx
        r1xsq   = r1x**2
        r2x     = isnx*r1xsq*ddnx/denx

        dpmlz[:nPML,:] = (np.arange(nPML, 0, -1)*dz).reshape((nPML,1))
        dpmlz[-nPML:,:] = (np.arange(1, nPML+1, 1)*dz).reshape((nPML,1))
        dnz     = pmlfz*c*dpmlz**2
        ddnz    = 2*pmlfz*c*dpmlz
        denz    = dnz + iom
        r1z     = iom / denz
        r1zsq   = r1z**2
        r2z     = isnz*r1zsq*ddnz/denz

        #For now, assume r1x and r1z are the same as xi_x and xi_z from operto et al. (2009)

        #pad edges
        r1x     = np.pad(r1x, pad_width=1, mode='edge')
        r1z     = np.pad(r1z, pad_width=1, mode='edge')

        Xi_x     = 1. / r1x[1,:].reshape((1,nx+2))
        Xi_z     = 1. / r1z[:,1].reshape((nz+2,1))

        # Visual key for finite-difference terms
        # (per Pratt and Worthington, 1990)
        #
        #   This         Original
        # AA BB CC  vs.  AD DD CD
        # DD EE FF  vs.  AA BE CC
        # GG HH II  vs.  AF FF CF

        # Set of keys to index the dictionaries

        # Anisotropic Stencil is 4 times the size, so we define 4 quadrants
        #
        # A =  M1 M2
        #      M3 M4

        # Diagonal offsets for the sparse matrix formation

        offsets = {
                'GG':   (-1) * nx + (-1),
                'HH':   (-1) * nx + ( 0),
                'II':   (-1) * nx + (+1),
                'DD':   ( 0) * nx + (-1),
                'EE':   ( 0) * nx + ( 0),
                'FF':   ( 0) * nx + (+1),
                'AA':   (+1) * nx + (-1),
                'BB':   (+1) * nx + ( 0),
                'CC':   (+1) * nx + (+1),
        }

        # Need to initialize the PML values

        Xi_x1 = Xi_x[:,0:-2] #left
        Xi_x2 = Xi_x[:,1:-1] #middle
        Xi_x3 = Xi_x[:,2:  ]   #right

        Xi_z1= Xi_z[0:-2,:] #left
        Xi_z2= Xi_z[1:-1,:] #middle
        Xi_z3= Xi_z[2:  ,:] #right

        # Here we will use the following notation
        #

        # Xi_x_M = (Xi_x(i)+Xi_(i-1))/2 --- M = 'minus'
        # Xi_x_C = (Xi_x(i)             --- C = 'centre'
        # Xi_x_P = (Xi_x(i)+Xi_(i+1))/2 --- P = 'plus'

        Xi_x_M = (Xi_x1+Xi_x2) / 2
        Xi_x_C = (Xi_x2)
        Xi_x_P = (Xi_x2+Xi_x3) / 2

        Xi_z_M = (Xi_z1+Xi_z2) / 2
        Xi_z_C = (Xi_z2)
        Xi_z_P = (Xi_z2+Xi_z3) / 2

        # Define Laplacian terms to shorten Stencil eqns

        L_x4 = 1 / (4*Xi_x_C*dxx)
        L_x = 1 / (Xi_x_C*dxx)

        L_z4 = 1 / (4*Xi_z_C*dzz)
        L_z = 1 / (Xi_z_C*dzz)

        # Buoyancies
        b_GG = 1. / rhoPad[0:-2,0:-2] # bottom left
        b_HH = 1. / rhoPad[0:-2,1:-1] # bottom centre
        b_II = 1. / rhoPad[0:-2,2:  ] # bottom right
        b_DD = 1. / rhoPad[1:-1,0:-2] # middle left
        b_EE = 1. / rhoPad[1:-1,1:-1] # middle centre
        b_FF = 1. / rhoPad[1:-1,2:  ] # middle right
        b_AA = 1. / rhoPad[2:  ,0:-2] # top    left
        b_BB = 1. / rhoPad[2:  ,1:-1] # top    centre
        b_CC = 1. / rhoPad[2:  ,2:  ] # top    right


        # Initialize averaged buoyancies on most of the grid

        # Here we will use the convention of 'sq' to represent the averaged bouyancy over 4 grid points,
        # and 'ln' to represent the bouyancy over 2 grid points:

        # SQ1 = AA BB        SQ2 =     BB CC        SQ3 = DD EE        SQ4 = EE FF
        #       DD EE                   EE FF              GG HH              HH II

        # LN1 = BB        LN2 = DD EE        LN3 = EE FF        LN4 = EE
        #       EE                                                              HH

        # We also introduce the suffixes 'x' and 'z' to
        # the averaged bouyancy squares to distinguish between
        # the x and z components with repsect to the PML decay
        # This is done, as before, to decrease the length of the stencil terms

        # Squares

        b_SQ1_x = ((b_AA + b_BB + b_DD + b_EE) / 4) / Xi_x_M
        b_SQ2_x = ((b_BB + b_CC + b_EE + b_FF) / 4) / Xi_x_P
        b_SQ3_x = ((b_DD + b_EE + b_GG + b_HH) / 4) / Xi_x_M
        b_SQ4_x = ((b_EE + b_FF + b_HH + b_II) / 4) / Xi_x_P

        b_SQ1_z = ((b_AA + b_BB + b_DD + b_EE) / 4) / Xi_z_M
        b_SQ2_z = ((b_BB + b_CC + b_EE + b_FF) / 4) / Xi_z_M
        b_SQ3_z = ((b_DD + b_EE + b_GG + b_HH) / 4) / Xi_z_P
        b_SQ4_z = ((b_EE + b_FF + b_HH + b_II) / 4) / Xi_z_P

        # Lines

        # Lines are in 1D, so no PML dim required
        # We use the Suffix 'C' for those terms where PML is not
        # calulated

        b_LN1 = ((b_BB + b_EE) / 2) / Xi_z_M
        b_LN2 = ((b_DD + b_EE) / 2) / Xi_x_M
        b_LN3 = ((b_EE + b_FF) / 2) / Xi_x_P
        b_LN4 = ((b_EE + b_HH) / 2) / Xi_z_P

        b_LN1_C = ((b_BB + b_EE) / 2) / Xi_x_C
        b_LN2_C = ((b_DD + b_EE) / 2) / Xi_z_C
        b_LN3_C = ((b_EE + b_FF) / 2) / Xi_z_C
        b_LN4_C = ((b_EE + b_HH) / 2) / Xi_x_C


        # Model parameter M
        K = omega*omega.conjugate() / (rhoPad * cPad**2)

        # K = omega^2/(c^2 . rho)

        K_GG = K[0:-2,0:-2] # bottom left
        K_HH = K[0:-2,1:-1] # bottom centre
        K_II = K[0:-2,2:  ] # bottom centre
        K_DD = K[1:-1,0:-2] # middle left
        K_EE = K[1:-1,1:-1] # middle centre
        K_FF = K[1:-1,2:  ] # middle right
        K_AA = K[2:  ,0:-2] # top    left
        K_BB = K[2:  ,1:-1] # top    centre
        K_CC = K[2:  ,2:  ] # top    right

        # 9-point fd star

        wm1 = 0.6291844;
        wm2 = 0.3708126;
        w1 = 0.4258673;

        # Mass Averaging Term

        # From Operto et al.(2009), anti-limped mass is calculted from 9 ponts and applied
        # ONLY to the diagonal terms

        K_avg = (wm1*K_EE) + ((wm2/4)*(K_BB + K_DD + K_FF + K_HH)) + (((1-wm1-wm2)/4)*(K_AA + K_CC + K_GG + K_II))

        # For now, set eps and delta to be constant

        theta   = self.theta
        eps     = self.eps
        delta   = self.delta

        # Need to define Anisotropic Matrix coeffs as in OPerto et al. (2009)

        Ax = 1 + (2*delta)*((np.cos(theta))**2)
        Bx = (-1*delta)*np.sin(2*theta)
        Cx = (1+(2*delta))*(np.cos(theta)**2)
        Dx = (-1*(1+(2*delta)))*((np.sin(2*theta))/2)
        Ex = (2*(eps-delta))*(np.cos(theta)**2)
        Fx = (-1*(eps-delta))*(np.sin(2*theta))
        Gx = Ex
        Hx = Fx

        Az = Bx
        Bz = 1 + ((2*delta)*(np.sin(theta)**2))
        Cz = Dx
        Dz = (1+(2*delta))*(np.sin(theta))
        Ez = Fx
        Fz = (2*(eps-delta))*(np.sin(theta)**2)
        Gz = Fx
        Hz = Fz

        keys = ['GG', 'HH', 'II', 'DD', 'EE', 'FF', 'AA', 'BB', 'CC']

        M1_diagonals = {
            'GG':  w1
                    * (
                      (((     L_x4) * Ax) * (   b_SQ3_x))
                    + (((-1 * L_x4) * Bx) * (   b_SQ3_z))
                    + (((-1 * L_z4) * Az) * (   b_SQ3_x))
                    + (((     L_z4) * Bz) * (   b_SQ3_z))
                      )
                    + (1-w1)
                    * (
                      (((-1 * L_x4) * Bx) * (   b_LN2_C))
                    + (((     L_z4) * Az) * (   b_LN4_C))
                    ),
            'HH':  w1
                    * (
                      (((     L_x4) * Ax) * ( - b_SQ3_x - b_SQ4_x))
                    + (((     L_x4) * Bx) * ( - b_SQ3_z + b_SQ4_z))
                    + (((     L_z4) * Az) * (   b_SQ3_x - b_SQ4_x))
                    + (((     L_z4) * Bz) * (   b_SQ3_z + b_SQ4_z))
                      )
                    + (1-w1)
                    * (
                      (((     L_x4) * Bx) * ( - b_LN2_C + b_LN3_C))
                    + (((      L_z) * Bz) * (   b_LN4))
                    ),
            'II':  w1
                    * (
                      (((     L_x4) * Ax) * (   b_SQ4_x))
                    + (((     L_x4) * Bx) * (   b_SQ4_z))
                    + (((     L_z4) * Az) * (   b_SQ4_x))
                    + (((     L_z4) * Bz) * (   b_SQ4_z))
                      )
                    + (1-w1)
                    * (
                      (((     L_x4) * Bx) * (   b_LN3_C))
                    + (((     L_z4) * Az) * (   b_LN4_C))
                    ),
            'DD':  w1
                    * (
                      (((     L_x4) * Ax) * (   b_SQ3_x + b_SQ1_x))
                    + (((     L_x4) * Bx) * (   b_SQ3_z - b_SQ1_z))
                    + (((     L_z4) * Az) * ( - b_SQ3_x + b_SQ1_x))
                    + (((     L_z4) * Bz) * ( - b_SQ3_z - b_SQ1_z))
                      )
                    + (1-w1)
                    * (
                      (((      L_x) * Ax) * (   b_LN2))
                    + (((     L_z4) * Az) * ( - b_LN4_C +  b_LN1_C))
                    ),
            'EE':  K_avg
                    + w1
                    * (
                      (((-1 * L_x4) * Ax) * (   b_SQ1_x + b_SQ2_x + b_SQ3_x + b_SQ4_x))
                    + (((     L_x4) * Bx) * (   b_SQ2_z + b_SQ3_z - b_SQ1_z - b_SQ4_z))
                    + (((     L_z4) * Az) * (   b_SQ2_x + b_SQ3_x - b_SQ1_x - b_SQ4_x))
                    + (((-1 * L_z4) * Bz) * (   b_SQ1_z + b_SQ2_z + b_SQ3_z + b_SQ4_z))
                      )
                    + (1-w1)
                    * (
                      (((      L_x) * Ax) * ( - b_LN2 - b_LN3))
                    + (((      L_z) * Bz) * ( - b_LN1 - b_LN4))
                      ),
            'FF':  w1
                    * (
                      (((     L_x4) * Ax) * (   b_SQ2_x + b_SQ4_x))
                    + (((     L_x4) * Bx) * (   b_SQ2_z - b_SQ4_z))
                    + (((     L_z4) * Az) * ( - b_SQ2_x + b_SQ4_x))
                    + (((     L_z4) * Bz) * ( - b_SQ2_z - b_SQ4_z))
                      )
                    + (1-w1)
                    * (
                      (((      L_x) * Ax) * (   b_LN3))
                    + (((     L_z4) * Az) * (   b_LN4_C - b_LN1_C))
                    ),
            'AA':  w1
                    * (
                      (((     L_x4) * Ax) * (   b_SQ1_x))
                    + (((     L_x4) * Bx) * (   b_SQ1_z))
                    + (((     L_z4) * Az) * (   b_SQ1_x))
                    + (((     L_z4) * Bz) * (   b_SQ1_z))
                      )
                    + (1-w1)
                    * (
                      (((-1 * L_x4) * Bx) * (   b_LN2_C))
                    + (((     L_z4) * Az) * (   b_LN1_C))
                    ),
            'BB':  w1
                    * (
                      (((     L_x4) * Ax) * ( - b_SQ2_x - b_SQ1_x))
                    + (((     L_x4) * Bx) * ( - b_SQ2_z + b_SQ1_z))
                    + (((     L_z4) * Az) * (   b_SQ2_x - b_SQ1_x))
                    + (((     L_z4) * Bz) * (   b_SQ2_z + b_SQ2_z))
                      )
                    + (1-w1)
                    * (
                      (((     L_x4) * Bx) * ( - b_LN3_C + b_LN2_C))
                    + (((      L_z) * Bz) * (   b_LN1))
                    ),
            'CC': w1
                    * (
                      (((     L_x4) * Ax) * (   b_SQ2_x))
                    + (((-1 * L_x4) * Bx) * (   b_SQ2_z))
                    + (((-1 * L_z4) * Az) * (   b_SQ2_x))
                    + (((     L_z4) * Bz) * (   b_SQ2_z))
                      )
                    + (1-w1)
                    * (
                      (((-1 * L_x4) * Bx) * (   b_LN3_C))
                    + (((-1 * L_z4) * Az) * (   b_LN1_C))
                    ),
        }

        M2_diagonals = {
            'GG':  w1
                    * (
                      (((     L_x4) * Cx) * (   b_SQ3_x))
                    + (((-1 * L_x4) * Dx) * (   b_SQ3_z))
                    + (((-1 * L_z4) * Cz) * (   b_SQ3_x))
                    + (((     L_z4) * Dz) * (   b_SQ3_z))
                      )
                    + (1-w1)
                    * (
                      (((-1 * L_x4) * Dx) * (   b_LN2_C))
                    + (((     L_z4) * Cz) * (   b_LN4_C))
                    ),
            'HH':  w1
                    * (
                      (((     L_x4) * Cx) * ( - b_SQ3_x - b_SQ4_x))
                    + (((     L_x4) * Dx) * ( - b_SQ3_z + b_SQ4_z))
                    + (((     L_z4) * Cz) * (   b_SQ3_x - b_SQ4_x))
                    + (((     L_z4) * Dz) * (   b_SQ3_z + b_SQ4_z))
                      )
                    + (1-w1)
                    * (
                      (((     L_x4) * Dx) * ( - b_LN2_C + b_LN3_C))
                    + (((      L_z) * Dz) * (   b_LN4))
                    ),
            'II':  w1
                    * (
                      (((     L_x4) * Cx) * (   b_SQ4_x))
                    + (((     L_x4) * Dx) * (   b_SQ4_z))
                    + (((     L_z4) * Cz) * (   b_SQ4_x))
                    + (((     L_z4) * Dz) * (   b_SQ4_z))
                      )
                    + (1-w1)
                    * (
                      (((     L_x4) * Dx) * (   b_LN3_C))
                    + (((     L_z4) * Cz) * (   b_LN4_C))
                    ),
            'DD':  w1
                    * (
                      (((     L_x4) * Cx) * (   b_SQ3_x + b_SQ1_x))
                    + (((     L_x4) * Dx) * (   b_SQ3_z - b_SQ1_z))
                    + (((     L_z4) * Cz) * ( - b_SQ3_x + b_SQ1_x))
                    + (((     L_z4) * Dz) * ( - b_SQ3_z - b_SQ1_z))
                      )
                    + (1-w1)
                    * (
                      (((      L_x) * Cx) * (   b_LN2))
                    + (((     L_z4) * Cz) * ( - b_LN4_C +  b_LN1_C))
                    ),
            'EE':
                    + w1
                    * (
                      (((-1 * L_x4) * Cx) * (   b_SQ1_x + b_SQ2_x + b_SQ3_x + b_SQ4_x))
                    + (((     L_x4) * Dx) * (   b_SQ2_z + b_SQ3_z - b_SQ1_z - b_SQ4_z))
                    + (((     L_z4) * Cz) * (   b_SQ2_x + b_SQ3_x - b_SQ1_x - b_SQ4_x))
                    + (((-1 * L_z4) * Dz) * (   b_SQ1_z + b_SQ2_z + b_SQ3_z + b_SQ4_z))
                      )
                    + (1-w1)
                    * (
                      (((      L_x) * Cx) * ( - b_LN2 - b_LN3))
                    + (((      L_z) * Dz) * ( - b_LN1 - b_LN4))
                      ),
            'FF':  w1
                    * (
                      (((     L_x4) * Cx) * (   b_SQ2_x + b_SQ4_x))
                    + (((     L_x4) * Dx) * (   b_SQ2_z - b_SQ4_z))
                    + (((     L_z4) * Cz) * ( - b_SQ2_x + b_SQ4_x))
                    + (((     L_z4) * Dz) * ( - b_SQ2_z - b_SQ4_z))
                      )
                    + (1-w1)
                    * (
                      (((      L_x) * Cx) * (   b_LN3))
                    + (((     L_z4) * Cz) * (   b_LN4_C - b_LN1_C))
                    ),
            'AA':  w1
                    * (
                      (((     L_x4) * Cx) * (   b_SQ1_x))
                    + (((     L_x4) * Dx) * (   b_SQ1_z))
                    + (((     L_z4) * Cz) * (   b_SQ1_x))
                    + (((     L_z4) * Dz) * (   b_SQ1_z))
                      )
                    + (1-w1)
                    * (
                      (((-1 * L_x4) * Dx) * (   b_LN2_C))
                    + (((     L_z4) * Cz) * (   b_LN1_C))
                    ),
            'BB':  w1
                    * (
                      (((     L_x4) * Cx) * ( - b_SQ2_x - b_SQ1_x))
                    + (((     L_x4) * Dx) * ( - b_SQ2_z + b_SQ1_z))
                    + (((     L_z4) * Cz) * (   b_SQ2_x - b_SQ1_x))
                    + (((     L_z4) * Dz) * (   b_SQ2_z + b_SQ2_z))
                      )
                    + (1-w1)
                    * (
                      (((     L_x4) * Dx) * ( - b_LN3_C + b_LN2_C))
                    + (((      L_z) * Dz) * (   b_LN1))
                    ),
            'CC': w1
                    * (
                      (((     L_x4) * Cx) * (   b_SQ2_x))
                    + (((-1 * L_x4) * Dx) * (   b_SQ2_z))
                    + (((-1 * L_z4) * Cz) * (   b_SQ2_x))
                    + (((     L_z4) * Dz) * (   b_SQ2_z))
                      )
                    + (1-w1)
                    * (
                      (((-1 * L_x4) * Dx) * (   b_LN3_C))
                    + (((-1 * L_z4) * Cz) * (   b_LN1_C))
                    ),
        }

        M3_diagonals = {
            'GG':  w1
                    * (
                      (((     L_x4) * Ex) * (   b_SQ3_x))
                    + (((-1 * L_x4) * Fx) * (   b_SQ3_z))
                    + (((-1 * L_z4) * Ez) * (   b_SQ3_x))
                    + (((     L_z4) * Fz) * (   b_SQ3_z))
                      )
                    + (1-w1)
                    * (
                      (((-1 * L_x4) * Fx) * (   b_LN2_C))
                    + (((     L_z4) * Ez) * (   b_LN4_C))
                    ),
            'HH':  w1
                    * (
                      (((     L_x4) * Ex) * ( - b_SQ3_x - b_SQ4_x))
                    + (((     L_x4) * Fx) * ( - b_SQ3_z + b_SQ4_z))
                    + (((     L_z4) * Ez) * (   b_SQ3_x - b_SQ4_x))
                    + (((     L_z4) * Fz) * (   b_SQ3_z + b_SQ4_z))
                      )
                    + (1-w1)
                    * (
                      (((     L_x4) * Fx) * ( - b_LN2_C + b_LN3_C))
                    + (((      L_z) * Fz) * (   b_LN4))
                    ),
            'II':  w1
                    * (
                      (((     L_x4) * Ex) * (   b_SQ4_x))
                    + (((     L_x4) * Fx) * (   b_SQ4_z))
                    + (((     L_z4) * Ez) * (   b_SQ4_x))
                    + (((     L_z4) * Fz) * (   b_SQ4_z))
                      )
                    + (1-w1)
                    * (
                      (((     L_x4) * Fx) * (   b_LN3_C))
                    + (((     L_z4) * Ez) * (   b_LN4_C))
                    ),
            'DD':  w1
                    * (
                      (((     L_x4) * Ex) * (   b_SQ3_x + b_SQ1_x))
                    + (((     L_x4) * Fx) * (   b_SQ3_z - b_SQ1_z))
                    + (((     L_z4) * Ez) * ( - b_SQ3_x + b_SQ1_x))
                    + (((     L_z4) * Fz) * ( - b_SQ3_z - b_SQ1_z))
                      )
                    + (1-w1)
                    * (
                      (((      L_x) * Ex) * (   b_LN2))
                    + (((     L_z4) * Ez) * ( - b_LN4_C +  b_LN1_C))
                    ),
            'EE':
                    + w1
                    * (
                      (((-1 * L_x4) * Ex) * (   b_SQ1_x + b_SQ2_x + b_SQ3_x + b_SQ4_x))
                    + (((     L_x4) * Fx) * (   b_SQ2_z + b_SQ3_z - b_SQ1_z - b_SQ4_z))
                    + (((     L_z4) * Ez) * (   b_SQ2_x + b_SQ3_x - b_SQ1_x - b_SQ4_x))
                    + (((-1 * L_z4) * Fz) * (   b_SQ1_z + b_SQ2_z + b_SQ3_z + b_SQ4_z))
                      )
                    + (1-w1)
                    * (
                      (((      L_x) * Ex) * ( - b_LN2 - b_LN3))
                    + (((      L_z) * Fz) * ( - b_LN1 - b_LN4))
                      ),
            'FF':  w1
                    * (
                      (((     L_x4) * Ex) * (   b_SQ2_x + b_SQ4_x))
                    + (((     L_x4) * Fx) * (   b_SQ2_z - b_SQ4_z))
                    + (((     L_z4) * Ez) * ( - b_SQ2_x + b_SQ4_x))
                    + (((     L_z4) * Fz) * ( - b_SQ2_z - b_SQ4_z))
                      )
                    + (1-w1)
                    * (
                      (((      L_x) * Ex) * (   b_LN3))
                    + (((     L_z4) * Ez) * (   b_LN4_C - b_LN1_C))
                    ),
            'AA':  w1
                    * (
                      (((     L_x4) * Ex) * (   b_SQ1_x))
                    + (((     L_x4) * Fx) * (   b_SQ1_z))
                    + (((     L_z4) * Ez) * (   b_SQ1_x))
                    + (((     L_z4) * Fz) * (   b_SQ1_z))
                      )
                    + (1-w1)
                    * (
                      (((-1 * L_x4) * Fx) * (   b_LN2_C))
                    + (((     L_z4) * Ez) * (   b_LN1_C))
                    ),
            'BB':  w1
                    * (
                      (((     L_x4) * Ex) * ( - b_SQ2_x - b_SQ1_x))
                    + (((     L_x4) * Fx) * ( - b_SQ2_z + b_SQ1_z))
                    + (((     L_z4) * Ez) * (   b_SQ2_x - b_SQ1_x))
                    + (((     L_z4) * Fz) * (   b_SQ2_z + b_SQ2_z))
                      )
                    + (1-w1)
                    * (
                      (((     L_x4) * Fx) * ( - b_LN3_C + b_LN2_C))
                    + (((      L_z) * Fz) * (   b_LN1))
                    ),
            'CC': w1
                    * (
                      (((     L_x4) * Ex) * (   b_SQ2_x))
                    + (((-1 * L_x4) * Fx) * (   b_SQ2_z))
                    + (((-1 * L_z4) * Ez) * (   b_SQ2_x))
                    + (((     L_z4) * Fz) * (   b_SQ2_z))
                      )
                    + (1-w1)
                    * (
                      (((-1 * L_x4) * Fx) * (   b_LN3_C))
                    + (((-1 * L_z4) * Ez) * (   b_LN1_C))
                    ),
        }

        M4_diagonals = {
            'GG':  w1
                    * (
                      (((     L_x4) * Gx) * (   b_SQ3_x))
                    + (((-1 * L_x4) * Hx) * (   b_SQ3_z))
                    + (((-1 * L_z4) * Gz) * (   b_SQ3_x))
                    + (((     L_z4) * Hz) * (   b_SQ3_z))
                      )
                    + (1-w1)
                    * (
                      (((-1 * L_x4) * Hx) * (   b_LN2_C))
                    + (((     L_z4) * Gz) * (   b_LN4_C))
                    ),
            'HH':  w1
                    * (
                      (((     L_x4) * Gx) * ( - b_SQ3_x - b_SQ4_x))
                    + (((     L_x4) * Hx) * ( - b_SQ3_z + b_SQ4_z))
                    + (((     L_z4) * Gz) * (   b_SQ3_x - b_SQ4_x))
                    + (((     L_z4) * Hz) * (   b_SQ3_z + b_SQ4_z))
                      )
                    + (1-w1)
                    * (
                      (((     L_x4) * Hx) * ( - b_LN2_C + b_LN3_C))
                    + (((      L_z) * Hz) * (   b_LN4))
                    ),
            'II':  w1
                    * (
                      (((     L_x4) * Gx) * (   b_SQ4_x))
                    + (((     L_x4) * Hx) * (   b_SQ4_z))
                    + (((     L_z4) * Gz) * (   b_SQ4_x))
                    + (((     L_z4) * Hz) * (   b_SQ4_z))
                      )
                    + (1-w1)
                    * (
                      (((     L_x4) * Hx) * (   b_LN3_C))
                    + (((     L_z4) * Gz) * (   b_LN4_C))
                    ),
            'DD':  w1
                    * (
                      (((     L_x4) * Gx) * (   b_SQ3_x + b_SQ1_x))
                    + (((     L_x4) * Hx) * (   b_SQ3_z - b_SQ1_z))
                    + (((     L_z4) * Gz) * ( - b_SQ3_x + b_SQ1_x))
                    + (((     L_z4) * Hz) * ( - b_SQ3_z - b_SQ1_z))
                      )
                    + (1-w1)
                    * (
                      (((      L_x) * Gx) * (   b_LN2))
                    + (((     L_z4) * Gz) * ( - b_LN4_C +  b_LN1_C))
                    ),
            'EE':  K_avg
                    + w1
                    * (
                      (((-1 * L_x4) * Gx) * (   b_SQ1_x + b_SQ2_x + b_SQ3_x + b_SQ4_x))
                    + (((     L_x4) * Hx) * (   b_SQ2_z + b_SQ3_z - b_SQ1_z - b_SQ4_z))
                    + (((     L_z4) * Gz) * (   b_SQ2_x + b_SQ3_x - b_SQ1_x - b_SQ4_x))
                    + (((-1 * L_z4) * Hz) * (   b_SQ1_z + b_SQ2_z + b_SQ3_z + b_SQ4_z))
                      )
                    + (1-w1)
                    * (
                      (((      L_x) * Gx) * ( - b_LN2 - b_LN3))
                    + (((      L_z) * Hz) * ( - b_LN1 - b_LN4))
                      ),
            'FF':  w1
                    * (
                      (((     L_x4) * Gx) * (   b_SQ2_x + b_SQ4_x))
                    + (((     L_x4) * Hx) * (   b_SQ2_z - b_SQ4_z))
                    + (((     L_z4) * Gz) * ( - b_SQ2_x + b_SQ4_x))
                    + (((     L_z4) * Hz) * ( - b_SQ2_z - b_SQ4_z))
                      )
                    + (1-w1)
                    * (
                      (((      L_x) * Gx) * (   b_LN3))
                    + (((     L_z4) * Gz) * (   b_LN4_C - b_LN1_C))
                    ),
            'AA':  w1
                    * (
                      (((     L_x4) * Gx) * (   b_SQ1_x))
                    + (((     L_x4) * Hx) * (   b_SQ1_z))
                    + (((     L_z4) * Gz) * (   b_SQ1_x))
                    + (((     L_z4) * Hz) * (   b_SQ1_z))
                      )
                    + (1-w1)
                    * (
                      (((-1 * L_x4) * Hx) * (   b_LN2_C))
                    + (((     L_z4) * Gz) * (   b_LN1_C))
                    ),
            'BB':  w1
                    * (
                      (((     L_x4) * Gx) * ( - b_SQ2_x - b_SQ1_x))
                    + (((     L_x4) * Hx) * ( - b_SQ2_z + b_SQ1_z))
                    + (((     L_z4) * Gz) * (   b_SQ2_x - b_SQ1_x))
                    + (((     L_z4) * Hz) * (   b_SQ2_z + b_SQ2_z))
                      )
                    + (1-w1)
                    * (
                      (((     L_x4) * Hx) * ( - b_LN3_C + b_LN2_C))
                    + (((      L_z) * Hz) * (   b_LN1))
                    ),
            'CC': w1
                    * (
                      (((     L_x4) * Gx) * (   b_SQ2_x))
                    + (((-1 * L_x4) * Hx) * (   b_SQ2_z))
                    + (((-1 * L_z4) * Gz) * (   b_SQ2_x))
                    + (((     L_z4) * Hz) * (   b_SQ2_z))
                      )
                    + (1-w1)
                    * (
                      (((-1 * L_x4) * Hx) * (   b_LN3_C))
                    + (((-1 * L_z4) * Gz) * (   b_LN1_C))
                    ),
        }

        # self._setupBoundary(diagonals, freeSurf)
        offsets = [offsets[key] for key in keys]

        M1_diagonals['GG'] = M1_diagonals['GG'].ravel()[nx+1:     ]
        M1_diagonals['HH'] = M1_diagonals['HH'].ravel()[nx  :     ]
        M1_diagonals['II'] = M1_diagonals['II'].ravel()[nx-1:     ]
        M1_diagonals['DD'] = M1_diagonals['DD'].ravel()[   1:     ]
        M1_diagonals['EE'] = M1_diagonals['EE'].ravel()[    :     ]
        M1_diagonals['FF'] = M1_diagonals['FF'].ravel()[    :-1   ]
        M1_diagonals['AA'] = M1_diagonals['AA'].ravel()[    :-nx+1]
        M1_diagonals['BB'] = M1_diagonals['BB'].ravel()[    :-nx  ]
        M1_diagonals['CC'] = M1_diagonals['CC'].ravel()[    :-nx-1]

        M1_diagonals = [M1_diagonals[key] for key in keys]

        M1_A = scipy.sparse.diags(M1_diagonals, offsets, shape=(nrows, nrows), format='csr', dtype=np.complex128)

        M2_diagonals['GG'] = M2_diagonals['GG'].ravel()[nx+1:     ]
        M2_diagonals['HH'] = M2_diagonals['HH'].ravel()[nx  :     ]
        M2_diagonals['II'] = M2_diagonals['II'].ravel()[nx-1:     ]
        M2_diagonals['DD'] = M2_diagonals['DD'].ravel()[   1:     ]
        M2_diagonals['EE'] = M2_diagonals['EE'].ravel()[    :     ]
        M2_diagonals['FF'] = M2_diagonals['FF'].ravel()[    :-1   ]
        M2_diagonals['AA'] = M2_diagonals['AA'].ravel()[    :-nx+1]
        M2_diagonals['BB'] = M2_diagonals['BB'].ravel()[    :-nx  ]
        M2_diagonals['CC'] = M2_diagonals['CC'].ravel()[    :-nx-1]

        M2_diagonals = [M2_diagonals[key] for key in keys]

        M2_A = scipy.sparse.diags(M2_diagonals, offsets, shape=(nrows, nrows), format='csr', dtype=np.complex128)

        M3_diagonals['GG'] = M3_diagonals['GG'].ravel()[nx+1:     ]
        M3_diagonals['HH'] = M3_diagonals['HH'].ravel()[nx  :     ]
        M3_diagonals['II'] = M3_diagonals['II'].ravel()[nx-1:     ]
        M3_diagonals['DD'] = M3_diagonals['DD'].ravel()[   1:     ]
        M3_diagonals['EE'] = M3_diagonals['EE'].ravel()[    :     ]
        M3_diagonals['FF'] = M3_diagonals['FF'].ravel()[    :-1   ]
        M3_diagonals['AA'] = M3_diagonals['AA'].ravel()[    :-nx+1]
        M3_diagonals['BB'] = M3_diagonals['BB'].ravel()[    :-nx  ]
        M3_diagonals['CC'] = M3_diagonals['CC'].ravel()[    :-nx-1]

        M3_diagonals = [M3_diagonals[key] for key in keys]

        M3_A = scipy.sparse.diags(M3_diagonals, offsets, shape=(nrows, nrows), format='csr', dtype=np.complex128)

        M4_diagonals['GG'] = M4_diagonals['GG'].ravel()[nx+1:     ]
        M4_diagonals['HH'] = M4_diagonals['HH'].ravel()[nx  :     ]
        M4_diagonals['II'] = M4_diagonals['II'].ravel()[nx-1:     ]
        M4_diagonals['DD'] = M4_diagonals['DD'].ravel()[   1:     ]
        M4_diagonals['EE'] = M4_diagonals['EE'].ravel()[    :     ]
        M4_diagonals['FF'] = M4_diagonals['FF'].ravel()[    :-1   ]
        M4_diagonals['AA'] = M4_diagonals['AA'].ravel()[    :-nx+1]
        M4_diagonals['BB'] = M4_diagonals['BB'].ravel()[    :-nx  ]
        M4_diagonals['CC'] = M4_diagonals['CC'].ravel()[    :-nx-1]

        M4_diagonals = [M4_diagonals[key] for key in keys]

        M4_A = scipy.sparse.diags(M4_diagonals, offsets, shape=(nrows, nrows), format='csr', dtype=np.complex128)

        # Need to switch these matrices together
        # A = [M1_A M2_A
        #      M3_A M4_A]

        top = scipy.sparse.hstack((M1_A,M2_A))
        bottom = scipy.sparse.hstack((M3_A,M4_A))

        A = scipy.sparse.vstack((top,bottom))
        return A

    def _setupBoundary(self, diagonals, freeSurf):
        """
        Function to set up boundary regions for the Seismic FDFD problem
        using the 9-point finite-difference stencil from OMEGA/FULLWV.
        """

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
                diagonals[key][0,:] = pickDiag(0)
            else:
                diagonals[key][0,:] = 0.

        # Top
        for key in keys:
            if key is 'BE':
                diagonals[key][-1,:] = pickDiag(2)
            else:
                diagonals[key][-1,:] = 0.

    @property
    def A(self):
        if getattr(self, '_A', None) is None:
            self._A = self._initHelmholtzNinePoint()
        return self._A

    @property
    def Solver(self):
        if getattr(self, '_Solver', None) is None:
            A = self.A.tocsc()
            self._Solver = scipy.sparse.linalg.splu(A)
        return self._Solver

    @property
    def theta(self):
        if getattr(self, '_theta', None) is None:
            self._theta = 0.5 * np.pi * np.ones((self.nz, self.nx))
        return self._theta

    @property
    def eps(self):
        if getattr(self, '_eps', None) is None:
            self._eps = np.zeros((self.nz, self.nx))
        return self._eps

    @property
    def delta(self):
        if getattr(self, '_delta', None) is None:
            self._delta = np.zeros((self.nz, self.nx))
        return self._delta

    def __mul__(self, value):
        u = self.Solver.solve(value)
        return u

def CB_mfact():

    global Ainv

    c = np.sqrt(fullwv.m[:fullwv.nz[0],:fullwv.nx[0]]/fullwv.rho[:fullwv.nz[0],:fullwv.nx[0]]).conjugate()
    freq = fullwv.omega[0].conjugate() / (2*np.pi)
    ky = fullwv.keiy[0] / (2*np.pi)

    newSystem = True
    if 'Ainv' in globals():
        checks = [
            not (Ainv.c - c).sum() == 0,
            not (Ainv.freq - freq).sum() == 0,
            not (Ainv.ky - ky).sum() == 0,
        ]
        newSystem = any(checks)

    if newSystem:
        fullwv.dbprint('Factorizing system')

        systemConfig = {
            'dx':       fullwv.dx[0],
            'dz':       fullwv.dz[0],
            'c':        c,
            'rho':      fullwv.rho[:fullwv.nz,:fullwv.nx],
            'nx':       fullwv.nx,
            'nz':       fullwv.nz,
            'freeSurf': fullwv.freesurf,
            'nPML':     10,
            'freq':     freq,
            'ky':       ky,
        }

        Ainv = Eurus(systemConfig)

def CB_sveq():

    q1 = fullwv.sfld[:fullwv.nx*fullwv.nz].reshape((fullwv.nx,fullwv.nz)).T.ravel()

    # for now, assume fullwv does it by source, so dimensions of q needs to be (nx*nz,1)
    q2=np.zeros(fullwv.nx*fullwv.nz,1)
    q = np.vstack((q1,q2))

    u = (Ainv*q).conjugate() * np.exp(-2j*np.pi* (0.25 + 0.006*Ainv.freq))
    fullwv.sfld[:fullwv.nx*fullwv.nz] = u.reshape((fullwv.nz,fullwv.nx)).T.ravel()
