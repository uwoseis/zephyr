
from anemoi.meta import BaseDiscretization

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

class Eurus(BaseDiscretization):
    
    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'c':            (True,      '_c',           np.complex128),
        'rho':          (False,     '_rho',         np.float64),
        'nPML':         (False,     '_nPML',        np.int64),
        'freq':         (True,      None,           np.complex128),
        'dx':           (True,      None,           np.float64),
        'dz':           (True,      None,           np.float64),
        'nx':           (True,      None,           np.int64),
        'nz':           (True,      None,           np.int64),
        'freeSurf':     (False,     None,           list),
        'mord':         (False,     '_mord',        tuple),
        'theta':        (False,     '_theta',       np.float64),
        'eps':          (False,     '_eps',         np.float64),
        'delta':        (False,     '_delta',       np.float64),
        'cPML':         (False,     '_cPML',        np.float64),
        'tau':          (False,     '_tau',         np.float64),
    }

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

        exec 'nf = %s'%self.mord[0] in locals()
        exec 'ns = %s'%self.mord[1] in locals()

        # fast --> slow is x --> y --> z as Fortran

        # Set up physical properties in matrices with padding
        omega   = 2*np.pi * self.freq
        padopts = {'pad_width': 1, 'mode': 'edge'}
        cPad    = np.pad(c.real, **padopts) + 1j * np.pad(c.imag, **padopts)
        rhoPad  = np.pad(rho, **padopts)

        # Horizontal, vertical and diagonal geometry terms
        dx  = self.dx
        dz  = self.dz
        dxx = dx**2
        dzz = dz**2
        dxz = (dxx+dzz)/2
        dd  = np.sqrt(dxz)
        omegaDamped = omega + self.dampcoeff

        # PML decay terms
        # NB: Arrays are padded later, but 'c' in these lines
        #     comes from the original (un-padded) version

        nPML    = self.nPML

        #Operto et al.(2009) PML implementation taken from Hudstedt et al.(2004)
        pmldx   = dx*(nPML - 1)
        pmldz   = dz*(nPML - 1)
        c_PML   = self.cPML

        gamma_x = np.zeros(nx, dtype=np.complex128)
        gamma_z = np.zeros(nz, dtype=np.complex128)

        x_vals  = np.arange(0,pmldx+dx,dx)
        z_vals  = np.arange(0,pmldz+dz,dz)

        gamma_x[:nPML]  = c_PML * (np.cos((np.pi/2)* (x_vals/pmldx)))
        gamma_x[-nPML:] = c_PML * (np.cos((np.pi/2)* (x_vals[::-1]/pmldx)))

        gamma_z[:nPML]  = c_PML * (np.cos((np.pi/2)* (z_vals/pmldz)))
        gamma_z[-nPML:] = c_PML * (np.cos((np.pi/2)* (z_vals[::-1]/pmldz)))

        gamma_x = np.pad(gamma_x.real, **padopts) + 1j * np.pad(gamma_x.imag, **padopts)
        gamma_z = np.pad(gamma_z.real, **padopts) + 1j * np.pad(gamma_z.imag, **padopts)

        Xi_x     = 1 + ((1j *gamma_x.reshape((1,nx+2)))/omegaDamped)
        Xi_z     = 1 + ((1j *gamma_z.reshape((nz+2,1)))/omegaDamped)


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
            'GG':   -nf -ns,
            'HH':   -nf    ,
            'II':   -nf +ns,
            'DD':       -ns,
            'EE':         0,
            'FF':       +ns,
            'AA':   +nf -ns,
            'BB':   +nf    ,
            'CC':   +nf +ns,
        }

        def prepareDiagonals(diagonals):
            for key in diagonals:
                diagonals[key] = diagonals[key].ravel()
                if offsets[key] < 0:
                    diagonals[key] = diagonals[key][-offsets[key]:]
                elif offsets[key] > 0:
                    diagonals[key] = diagonals[key][:-offsets[key]]
                diagonals[key] = diagonals[key].ravel()

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
        K = (omegaDamped**2) / (rhoPad * cPad**2)

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

        def generateDiagonals(massTerm, coeff1x, coeff1z, coeff2x, coeff2z):

            diagonals = {
                'GG':  w1
                        * (
                          (((     L_x4) * coeff1x) * (   b_SQ3_x))
                        + (((-1 * L_x4) * coeff2x) * (   b_SQ3_z))
                        + (((-1 * L_z4) * coeff1z) * (   b_SQ3_x))
                        + (((     L_z4) * coeff2z) * (   b_SQ3_z))
                          )
                        + (1-w1)
                        * (
                          (((-1 * L_x4) * coeff2x) * (   b_LN2_C))
                        + (((     L_z4) * coeff1z) * (   b_LN4_C))
                        ),
                'HH':  w1
                        * (
                          (((     L_x4) * coeff1x) * ( - b_SQ3_x - b_SQ4_x))
                        + (((     L_x4) * coeff2x) * ( - b_SQ3_z + b_SQ4_z))
                        + (((     L_z4) * coeff1z) * (   b_SQ3_x - b_SQ4_x))
                        + (((     L_z4) * coeff2z) * (   b_SQ3_z + b_SQ4_z))
                          )
                        + (1-w1)
                        * (
                          (((     L_x4) * coeff2x) * ( - b_LN2_C + b_LN3_C))
                        + (((      L_z) * coeff2z) * (   b_LN4))
                        ),
                'II':  w1
                        * (
                          (((     L_x4) * coeff1x) * (   b_SQ4_x))
                        + (((     L_x4) * coeff2x) * (   b_SQ4_z))
                        + (((     L_z4) * coeff1z) * (   b_SQ4_x))
                        + (((     L_z4) * coeff2z) * (   b_SQ4_z))
                          )
                        + (1-w1)
                        * (
                          (((     L_x4) * coeff2x) * (   b_LN3_C))
                        + (((     L_z4) * coeff1z) * (   b_LN4_C))
                        ),
                'DD':  w1
                        * (
                          (((     L_x4) * coeff1x) * (   b_SQ3_x + b_SQ1_x))
                        + (((     L_x4) * coeff2x) * (   b_SQ3_z - b_SQ1_z))
                        + (((     L_z4) * coeff1z) * ( - b_SQ3_x + b_SQ1_x))
                        + (((     L_z4) * coeff2z) * ( - b_SQ3_z - b_SQ1_z))
                          )
                        + (1-w1)
                        * (
                          (((      L_x) * coeff1x) * (   b_LN2))
                        + (((     L_z4) * coeff1z) * ( - b_LN4_C +  b_LN1_C))
                        ),
                'EE':  massTerm
                        + w1
                        * (
                          (((-1 * L_x4) * coeff1x) * (   b_SQ1_x + b_SQ2_x + b_SQ3_x + b_SQ4_x))
                        + (((     L_x4) * coeff2x) * (   b_SQ2_z + b_SQ3_z - b_SQ1_z - b_SQ4_z))
                        + (((     L_z4) * coeff1z) * (   b_SQ2_x + b_SQ3_x - b_SQ1_x - b_SQ4_x))
                        + (((-1 * L_z4) * coeff2z) * (   b_SQ1_z + b_SQ2_z + b_SQ3_z + b_SQ4_z))
                          )
                        + (1-w1)
                        * (
                          (((      L_x) * coeff1x) * ( - b_LN2 - b_LN3))
                        + (((      L_z) * coeff2z) * ( - b_LN1 - b_LN4))
                          ),
                'FF':  w1
                        * (
                          (((     L_x4) * coeff1x) * (   b_SQ2_x + b_SQ4_x))
                        + (((     L_x4) * coeff2x) * (   b_SQ2_z - b_SQ4_z))
                        + (((     L_z4) * coeff1z) * ( - b_SQ2_x + b_SQ4_x))
                        + (((     L_z4) * coeff2z) * ( - b_SQ2_z - b_SQ4_z))
                          )
                        + (1-w1)
                        * (
                          (((      L_x) * coeff1x) * (   b_LN3))
                        + (((     L_z4) * coeff1z) * (   b_LN4_C - b_LN1_C))
                        ),
                'AA':  w1
                        * (
                          (((     L_x4) * coeff1x) * (   b_SQ1_x))
                        + (((     L_x4) * coeff2x) * (   b_SQ1_z))
                        + (((     L_z4) * coeff1z) * (   b_SQ1_x))
                        + (((     L_z4) * coeff2z) * (   b_SQ1_z))
                          )
                        + (1-w1)
                        * (
                          (((-1 * L_x4) * coeff2x) * (   b_LN2_C))
                        + (((     L_z4) * coeff1z) * (   b_LN1_C))
                        ),
                'BB':  w1
                        * (
                          (((     L_x4) * coeff1x) * ( - b_SQ2_x - b_SQ1_x))
                        + (((     L_x4) * coeff2x) * ( - b_SQ2_z + b_SQ1_z))
                        + (((     L_z4) * coeff1z) * (   b_SQ2_x - b_SQ1_x))
                        + (((     L_z4) * coeff2z) * (   b_SQ2_z + b_SQ2_z))
                          )
                        + (1-w1)
                        * (
                          (((     L_x4) * coeff2x) * ( - b_LN3_C + b_LN2_C))
                        + (((      L_z) * coeff2z) * (   b_LN1))
                        ),
                'CC': w1
                        * (
                          (((     L_x4) * coeff1x) * (   b_SQ2_x))
                        + (((-1 * L_x4) * coeff2x) * (   b_SQ2_z))
                        + (((-1 * L_z4) * coeff1z) * (   b_SQ2_x))
                        + (((     L_z4) * coeff2z) * (   b_SQ2_z))
                          )
                        + (1-w1)
                        * (
                          (((-1 * L_x4) * coeff2x) * (   b_LN3_C))
                        + (((-1 * L_z4) * coeff1z) * (   b_LN1_C))
                        ),
            }

            return diagonals

        M1_diagonals = generateDiagonals(K_avg, Ax, Az, Bx, Bz)
        prepareDiagonals(M1_diagonals)

        M2_diagonals = generateDiagonals(0.   , Gx, Gz, Hx, Hz)
        prepareDiagonals(M2_diagonals)

        M3_diagonals = generateDiagonals(0.   , Ex, Ez, Fx, Fz)
        prepareDiagonals(M3_diagonals)

        M4_diagonals = generateDiagonals(K_avg, Cx, Cz, Dx, Dz)
        prepareDiagonals(M4_diagonals)

        # self._setupBoundary(diagonals, freeSurf)
        offsets = [offsets[key] for key in keys]

        M1_diagonals = [M1_diagonals[key] for key in keys]
        M1_A = scipy.sparse.diags(M1_diagonals, offsets, shape=(nrows, nrows), format='csr', dtype=np.complex128)

        M2_diagonals = [M2_diagonals[key] for key in keys]
        M2_A = scipy.sparse.diags(M2_diagonals, offsets, shape=(nrows, nrows), format='csr', dtype=np.complex128)

        M3_diagonals = [M3_diagonals[key] for key in keys]
        M3_A = scipy.sparse.diags(M3_diagonals, offsets, shape=(nrows, nrows), format='csr', dtype=np.complex128)

        M4_diagonals = [M4_diagonals[key] for key in keys]
        M4_A = scipy.sparse.diags(M4_diagonals, offsets, shape=(nrows, nrows), format='csr', dtype=np.complex128)

        # Need to switch these matrices together
        # A = [M1_A M2_A
        #      M3_A M4_A]

        top = scipy.sparse.hstack((M1_A,M2_A))
        bottom = scipy.sparse.hstack((M3_A,M4_A))

        A = scipy.sparse.vstack((top,bottom))
        return A

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
    def mord(self):
        return getattr(self, '_mord', ('+nx', '+1'))

    @property
    def cPML(self):
        return getattr(self, '_cPML', 1e3)

    @property
    def nPML(self):
        return getattr(self, '_nPML', 10)
    
    @property
    def c(self):
        if isinstance(self._c, np.ndarray):
            return self._c
        else:
            return self._c * np.ones((self.nz, self.nx), dtype=np.complex128)
        
    @property
    def rho(self):
        if getattr(self, '_rho', None) is None:
            self._rho = 310. * self.c**0.25 
            
        if isinstance(self._rho, np.ndarray):
            return self._rho
        else:
            return self._rho * np.ones((self.nz, self.nx), dtype=np.float64)

    @property
    def theta(self):
        if getattr(self, '_theta', None) is None:
            self._theta = np.zeros((self.nz, self.nx))
            
        if isinstance(self._theta, np.ndarray):
            return self._theta
        else:
            return self._theta * np.ones((self.nz, self.nx), dtype=np.float64)

    @property
    def eps(self):
        if getattr(self, '_eps', None) is None:
            self._eps = np.zeros((self.nz, self.nx))
            
        if isinstance(self._eps, np.ndarray):
            return self._eps
        else:
            return self._eps * np.ones((self.nz, self.nx), dtype=np.float64)

    @property
    def delta(self):
        if getattr(self, '_delta', None) is None:
            self._delta = np.zeros((self.nz, self.nx))
            
        if isinstance(self._delta, np.ndarray):
            return self._delta
        else:
            return self._delta * np.ones((self.nz, self.nx), dtype=np.float64)
    
    @property
    def dampcoeff(self):
        return 1j / getattr(self, '_tau', np.inf)

    def __mul__(self, value):
        u = self.Solver.solve(value)
        return u
    
    def __call__(self, value):
        u = self.Solver.solve(value)
        return u
