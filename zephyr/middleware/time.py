
import numpy as np

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

