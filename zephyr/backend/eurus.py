
from .meta import BaseAnisotropic
from .discretization import BaseDiscretization

import numpy as np
import scipy.sparse as sp

class Eurus(BaseDiscretization, BaseAnisotropic):
    '''
    Implements Transversely Isotropic 2D (visco)acoustic frequency-domain wave physics using a mixed-grid
    finite-difference approach (Originally Proposed by Operto et al. (2009)).
    '''

    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'nPML':         (False,     '_nPML',        np.int64),
        'freq':         (True,      None,           np.complex128),
        'mord':         (False,     '_mord',        tuple),
        'cPML':         (False,     '_cPML',        np.float64),
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

        c = self.c.reshape(dims)
        rho = self.rho.reshape(dims)

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
        dxx = dx**2.
        dzz = dz**2.
        dxz = (dxx+dzz)/2.
        dd  = np.sqrt(dxz)
        omegaDamped = omega - self.dampCoeff

        # PML decay terms
        # NB: Arrays are padded later, but 'c' in these lines
        #     comes from the original (un-padded) version

        nPML    = self.nPML

        #Operto et al.(2009) PML implementation taken from Hudstedt et al.(2004)
        pmldx   = dx*(nPML - 1)
        pmldz   = dz*(nPML - 1)
        cPML   = self.cPML

        gamma_x = np.zeros(nx, dtype=np.complex128)
        gamma_z = np.zeros(nz, dtype=np.complex128)

        x_vals  = np.arange(0,pmldx+dx,dx)
        z_vals  = np.arange(0,pmldz+dz,dz)

        gamma_x[:nPML]  = cPML * (np.cos((np.pi/2)* (x_vals/pmldx)))
        gamma_x[-nPML:] = cPML * (np.cos((np.pi/2)* (x_vals[::-1]/pmldx)))

        gamma_z[:nPML]  = cPML * (np.cos((np.pi/2)* (z_vals/pmldz)))
        gamma_z[-nPML:] = cPML * (np.cos((np.pi/2)* (z_vals[::-1]/pmldz)))

        gamma_x = np.pad(gamma_x.real, **padopts) + 1j * np.pad(gamma_x.imag, **padopts)
        gamma_z = np.pad(gamma_z.real, **padopts) + 1j * np.pad(gamma_z.imag, **padopts)

        Xi_x     = 1 - ((1j *gamma_x.reshape((1,nx+2)))/omegaDamped)
        Xi_z     = 1 - ((1j *gamma_z.reshape((nz+2,1)))/omegaDamped)


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
        K = (omegaDamped * omegaDamped) / (rhoPad * cPad**2)
        #K = (omega**2) / (rhoPad * cPad**2)

        # K = omega^2/(c^2 . rho)

        KGG = K[0:-2,0:-2] # bottom left
        KHH = K[0:-2,1:-1] # bottom centre
        KII = K[0:-2,2:  ] # bottom centre
        KDD = K[1:-1,0:-2] # middle left
        KEE = K[1:-1,1:-1] # middle centre
        KFF = K[1:-1,2:  ] # middle right
        KAA = K[2:  ,0:-2] # top    left
        KBB = K[2:  ,1:-1] # top    centre
        KCC = K[2:  ,2:  ] # top    right

        # 9-point fd star

        wm1 = 0.6287326
        wm2 = 0.3712667
        wm3 = 1.- wm1 -wm2
        wm2 = 0.25 * wm2
        wm3 = 0.25 * wm3

        w1 = 0.4382634
        #w1 = 0.
        # Mass Averaging Term

        # From Operto et al.(2009), anti-limped mass is calculted from 9 ponts
        #

        #K_avg = (wm1*K_EE) + ((wm2/4)*(K_BB + K_DD + K_FF + K_HH)) + (((1-wm1-wm2)/4)*(K_AA + K_CC + K_GG + K_II))

        KGG = wm3 * KGG
        KHH = wm2 * KHH
        KII = wm3 * KII
        KDD = wm2 * KDD
        KEE = wm1 * KEE
        KFF = wm2 * KFF
        KAA = wm3 * KAA
        KBB = wm2 * KBB
        KCC = wm3 * KCC

        # For now, set eps and delta to be constant

        theta   = self.theta
        eps     = self.eps
        delta   = self.delta

        # Need to define Anisotropic Matrix coeffs as in OPerto et al. (2009)

        Ax = 1. + (2.*delta)*(np.cos(theta)**2.)
        Bx = (-1.*delta)*np.sin(2.*theta)
        Cx = (1.+(2.*delta))*(np.cos(theta)**2.)
        Dx = (-0.5*(1.+(2.*delta)))*((np.sin(2.*theta)))
        Ex = (2.*(eps-delta))*(np.cos(theta)**2.)
        Fx = (-1.*(eps-delta))*(np.sin(2.*theta))
        Gx = Ex
        Hx = Fx

        Az = Bx
        Bz = 1. + (2.*delta)*(np.sin(theta)**2.)
        Cz = Dx
        Dz = (1.+(2.*delta))*(np.sin(theta)**2.)
        Ez = Fx
        Fz = (2.*(eps-delta))*(np.sin(theta)**2.)
        Gz = Fx
        Hz = Fz


        keys = ['GG', 'HH', 'II', 'DD', 'EE', 'FF', 'AA', 'BB', 'CC']

        def generateDiagonals(massTerm, coeff1x, coeff1z, coeff2x, coeff2z, KAA, KBB, KCC, KDD, KEE, KFF, KGG, KHH, KII):
            '''
            Generates the sparse diagonals that comprise the 9-point mixed-grid anisotropic stencil.

            See Appendix of Operto et a. (2009)
            '''

            diagonals = {
                'GG':  (massTerm * KGG)
                        + w1
                        * (
                          (((     L_x4) * coeff1x) * (   b_SQ3_x))
                        + (((-1 * L_x4) * coeff2x) * (   b_SQ3_z))
                        + (((-1 * L_z4) * coeff1z) * (   b_SQ3_x))
                        + (((     L_z4) * coeff2z) * (   b_SQ3_z))
                          )
                        + (1-w1)
                        * (
                          (((-1 * L_x4) * coeff2x) * (   b_LN2_C))
                        + (((-1 * L_z4) * coeff1z) * (   b_LN4_C))
                        ),
                'HH':  (massTerm * KHH)
                        + w1
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
                'II':  (massTerm * KII)
                        + w1
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
                'DD':   (massTerm * KDD)
                        + w1
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
                'EE':   (massTerm * KEE)
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
                'FF':  (massTerm * KFF)
                        + w1
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
                'AA':  (massTerm * KAA)
                        + w1
                        * (
                          (((     L_x4) * coeff1x) * (   b_SQ1_x))
                        + (((     L_x4) * coeff2x) * (   b_SQ1_z))
                        + (((     L_z4) * coeff1z) * (   b_SQ1_x))
                        + (((     L_z4) * coeff2z) * (   b_SQ1_z))
                          )
                        + (1-w1)
                        * (
                          (((     L_x4) * coeff2x) * (   b_LN2_C))
                        + (((     L_z4) * coeff1z) * (   b_LN1_C))
                        ),
                'BB':  (massTerm * KBB)
                        + w1
                        * (
                          (((     L_x4) * coeff1x) * ( - b_SQ2_x - b_SQ1_x))
                        + (((     L_x4) * coeff2x) * ( - b_SQ2_z + b_SQ1_z))
                        + (((     L_z4) * coeff1z) * (   b_SQ2_x - b_SQ1_x))
                        + (((     L_z4) * coeff2z) * (   b_SQ2_z + b_SQ1_z))
                          )
                        + (1-w1)
                        * (
                          (((     L_x4) * coeff2x) * ( - b_LN3_C + b_LN2_C))
                        + (((      L_z) * coeff2z) * (   b_LN1))
                        ),
                'CC': (massTerm * KCC)
                        + w1
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


        M1_diagonals = generateDiagonals(1., Ax, Az, Bx, Bz, KAA, KBB, KCC, KDD, KEE, KFF, KGG, KHH, KII)
        self._setupBoundary(M1_diagonals)
        prepareDiagonals(M1_diagonals)

        M2_diagonals = generateDiagonals(0. , Cx, Cz, Dx, Dz, KAA, KBB, KCC, KDD, KEE, KFF, KGG, KHH, KII)
        self._setupBoundary(M2_diagonals)
        prepareDiagonals(M2_diagonals)

        M3_diagonals = generateDiagonals(0. , Ex, Ez, Fx, Fz, KAA, KBB, KCC, KDD, KEE, KFF, KGG, KHH, KII)
        self._setupBoundary(M3_diagonals)
        prepareDiagonals(M3_diagonals)

        M4_diagonals = generateDiagonals(1. ,Gx, Gz, Hx, Hz,  KAA, KBB, KCC, KDD, KEE, KFF, KGG, KHH, KII)
        self._setupBoundary(M4_diagonals)
        prepareDiagonals(M4_diagonals)

        offsets = [offsets[key] for key in keys]

        M1_diagonals = [M1_diagonals[key] for key in keys]
        M1_A = sp.diags(M1_diagonals, offsets, shape=(nrows, nrows), format='csr', dtype=np.complex128)

        M2_diagonals = [M2_diagonals[key] for key in keys]
        M2_A = sp.diags(M2_diagonals, offsets, shape=(nrows, nrows), format='csr', dtype=np.complex128)

        M3_diagonals = [M3_diagonals[key] for key in keys]
        M3_A = sp.diags(M3_diagonals, offsets, shape=(nrows, nrows), format='csr', dtype=np.complex128)

        M4_diagonals = [M4_diagonals[key] for key in keys]
        M4_A = sp.diags(M4_diagonals, offsets, shape=(nrows, nrows), format='csr', dtype=np.complex128)

        # A = [M1_A M2_A
        #      M3_A M4_A]

        A = sp.bmat([[M1_A, M2_A],[M3_A,M4_A]])
        return A

    @staticmethod
    def _setupBoundary(diagonals):
        '''
        Function to set up boundary regions for the Seismic FDFD problem
        using the 9-point finite-difference stencil from OMEGA/FULLWV.

        Args:
            diagonals (dict): The diagonal vectors, indexed by appropriate string keys
            freeSurf (tuple): Determines which free-surface conditions are active

        The diagonals are modified in-place.
        '''

        keys = [key for key in diagonals if key is not 'EE']

        for key in keys:
            diagonals[key][:,0] = 0.
            diagonals[key][:,-1] = 0.
            diagonals[key][0,:] = 0.
            diagonals[key][-1,:] = 0.

    @property
    def A(self):
        'The sparse system matrix'
        if getattr(self, '_A', None) is None:
            self._A = self._initHelmholtzNinePoint()
        return self._A

    @property
    def mord(self):
        'Determines matrix ordering'

        return getattr(self, '_mord', ('-nx', '+1'))

    @property
    def cPML(self):
        'The convolutional PML coefficient. It is experimentally determined for each project.'

        return getattr(self, '_cPML', 1e3)

    @property
    def nPML(self):
        'The depth of the PML (Perfectly Matched Layer) region in gridpoints'

        return getattr(self, '_nPML', 10)

    def __mul__(self, rhs):
        'The action of the inverse of the matrix A'
        clipResult = False

        if 2*rhs.shape[0] == self.shape[1]:

            if isinstance(rhs, sp.spmatrix):
                rhs = sp.vstack([rhs, sp.csr_matrix(rhs.shape, dtype=np.complex128)])
            else:
                rhs = np.vstack([rhs, np.zeros(rhs.shape, dtype=np.complex128)])

            clipResult = True

        elif rhs.shape[0] != self.shape[1]:
            raise ValueError('dimension mismatch')

        result = super(Eurus, self).__mul__(rhs)

        if clipResult:
            result = result[:self.shape[1]/2,:]

        return result


class EurusHD(Eurus):
    '''
    Implements Transversely Isotropic 2D (visco)acoustic frequency-domain wave physics using a mixed-grid
    finite-difference approach (Originally Proposed by Operto et al. (2009)).

    Includes half-differentiation of the source by default.
    '''

    @property
    def premul(self):
        '''
        A premultiplication factor, used by 2.5D. The default value implements
        half-differentiation of the source, which corrects for 3D spreading.
        '''

        cfact = np.sqrt(2j*np.pi * self.freq)
        return getattr(self, '_premul', cfact)
