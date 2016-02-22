
import numpy as np
from ..backend import AttributeMapper

def dwavelet(srcfreq, deltat, nexc):
    '''
    DWAVELET Calculates derivative Keuper wavelet, given the sample interval

    Define frequency and number of excursions, then compute number of
    samples and then calculate the source wavelet

    Based on dwavelet.m by R.G. Pratt
    '''

    m = (int(nexc) + 2) / float(nexc)
    nsrc = int((1./srcfreq)/deltat)
    delta = nexc * np.pi * srcfreq

    tsrc = np.arange(0, nsrc*deltat, deltat)
    source = delta * (np.cos(delta*tsrc) - np.cos(m*delta*tsrc))

    return source

def dftreal(a, N, M):
    '''
    Calculates multiple 1D forward DFT from real to complex
    Only first N/2 samples (positive frequencies) are returned

    Input:
        a - input real vectors in column form (samples from zero to N-1)
        N - implied number of complex output samples (= number of input samples)
        M - number of input vectors

    Based on dftreal.m by R.G. Pratt
    '''

    A = np.zeros((np.fix(N/2), M))
    n = np.arange(N).reshape((N,1))
    nk = n.T * n
    w = np.exp(2j*np.pi / N)
    W = w**(nk)
    A = np.dot(W, a[:N,:M]) / N

    return A

def idftreal(A, N, M):
    '''
    Calculates multiple 1D inverse DFT from complex to real

    Input:
        A - input complex vectors in column form (samples from zero to Nyquist)
        N - number of output samples (= number of implied complex input samples)
        M - number of input vectors

    Based on idftreal.m by R.G. Pratt
    '''

    a = np.zeros((N, M))
    n = np.arange(N).reshape((N,1))

    # Set maximum non-Nyquist frequency index (works for even or odd N)
    imax = np.int(np.fix((N+1)/2)-1)
    k1 = np.arange(np.fix(N/2)+1)               # Freq indices from zero to Nyquist
    k2 = np.arange(1, imax+1)                   # Freq indices except zero and Nyquist
    nk1 = n * k1.T
    nk2 = n * k2.T
    w = np.exp(-2j*np.pi / N)
    W = w**nk1
    W2 = w**nk2
    W[:,1:imax+1] += W2                         # Add two matrices properly shifted
    a = np.dot(W, A[:np.fix(N/2)+1,:M]).real    # (leads to doubling for non-Nyquist)

    return a


class TimeMachine(AttributeMapper):

    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'tau':          (False,     None,           np.float64),
        'freqs':        (True,      None,           list),
        'dt':           (False,     None,           np.float64),
        'freqBase':     (False,     None,           np.float64),
    }

    # @classmethod
    # def freqsFromTimes(cls, )

    @property
    def dt(self):
        if not hasattr(self, '_dt'):
            self._dt = 1. / self.fMax
        return getattr(self, '_dt', )
    @dt.setter
    def dt(self, value):
        self._dt = value

    @property
    def tMax(self):
        return 1. / self.df

    @property
    def fMax(self):
        return self.freqs[-1]
    
    @property
    def df(self):
        if len(self.freqs) > 1:
            return self.freqs[1] - self.freqs[0]
        else:
            return 1.

    @property
    def nom(self):
        return len(self.freqs)

    @property
    def ns(self):
        return 2 * self.nom

    @property
    def freqs(self):
        return self._freqs
    @freqs.setter
    def freqs(self, value):

        if len(value) > 1:
            step = value[1] - value[0]
            for i in xrange(1, len(value)):
                ostep = step
                step = value[i] - value[i-1]
                if abs(step - ostep) > 1e-5:
                    raise Exception('%(class)s requires that the frequencies be sampled regularly'%{'class': self.__class__.__name__})

        self._freqs = value

    @property
    def freqBase(self):
        return getattr(self, '_freqBase', 0.)
    @freqBase.setter
    def freqBase(self, value):
        assert value >= 0
        self._freqBase = value

    @property
    def tau(self):
        'Laplace-domain damping time constant'
        return getattr(self, '_tau', np.inf)
    @tau.setter
    def tau(self, value):
        self._tau = value
    
    @property
    def dampCoeff(self):
        'Computed damping coefficient to be added to real omega'
        return 1j / self.tau 

    def keuper(self, freq=None, nexc=2, dt=None):
        '''
        Generate a Keuper wavelet.
        '''

        if freq is None:
            if not self.freqBase > 0.:
                raise TypeError('%(class)s requires argument \'freq\', unless it is determined from freqBase'%{'class': self.__class__.__name__})
            freq = self.freqBase

        if dt is None:
            dt = self.dt

        wavelet = dwavelet(freq, dt, nexc)
        tseries = np.zeros((self.ns,), dtype=np.float64)
        tseries[:len(wavelet)] = wavelet

        return tseries

    def fSource(self, tdata):
        '''
        Convert a time series source to equally-spaced frequencies.
        '''

        if tdata.ndim < 2:
            tdata = tdata.reshape((1, len(tdata)))
        fdata = self.dft(tdata)

        return fdata[:, 1:fdata.shape[1]/2 + 1]

    def dft(self, a):
        '''
        Automatically carry out the forward discrete Fourier transform.
        '''

        a = a.T

        return dftreal(a, a.shape[0], a.shape[1]).T


    def idft(self, A):
        '''
        Automatically carry out the inverse discrete Fourier transform.
        '''

        A = A.T
        ns = 2*A.shape[0]
        A = np.vstack([np.zeros((1, A.shape[1]), dtype=np.complex128), A])

        return idftreal(A, ns, A.shape[1]).T

    def fft(self, a):
        '''
        Automatically carry out the forward fast Fourier transform.
        '''
        
        raise NotImplementedError

    def ifft(self, A):
        '''
        Automatically carry out the inverse fast Fourier transform.
        '''
        
        raise NotImplementedError

    def timeSlice(self, slices):
        '''
        Carry out forward modelling and return time slices.
        '''

        raise NotImplementedError

