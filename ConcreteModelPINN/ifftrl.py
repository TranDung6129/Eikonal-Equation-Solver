import numpy as np
from scipy.fft import ifft

def ifftrl(spec, f):
    """
    Inverse Fourier transform for real-valued signal

    Inverse fourier transform of a real trace. The spectrum is assumed
    to be of length floor(n/2 + 1) where n is the length of the time series 
    to be created. (This is automatically the case if it was created by fftrl).

    Parameters:
    - spec (ndarray): The input spectrum
    - f (ndarray): The frequency coordinate vector

    Returns:
    - r: output trace
    - t: output time coordinate vector
    """
    # Test for matrix or vector
    [m, n] = spec.shape
    itr = 0
    if not (m - 1) * (n - 1):
        if m == 1:
            spec = spec.T
            itr = 1
        nsamp = len(spec)
        ntr = 1
    else:
        nsamp = m
        ntr = n
    
    # Form the conjugate symmetric complex spectrum expected by ifft
    # Test for presence of nyquist
    nyq = 0
    if np.isreal(spec[-1]):
        nyq = 1
    if nyq:
        L1 = np.arange(1, nsamp + 1)
        L2 = np.arange(nsamp -1, 1, -1)
    else:
        L1 = np.arange(1, nsamp + 1)
        L2 = np.arange(nsamp, 1, -1)

    symspec = np.vstack((spec[L1 - 1, :], np.conj(spec[L2 - 1, :])))

    # Transform the array
    r = np.real(ifft(symspec, axis=0))
    
    # Build the time vector
    n = len(r)
    df = f[1] - f[0]
    dt = 1 / (df * n)
    t = np.arange(0, n - 1) * dt

    if itr == 1:
        r = np.transpose(r)
        t = np.transpose(t)

    return r, t
    