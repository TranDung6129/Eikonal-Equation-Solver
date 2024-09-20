import numpy as np
from scipy.fft import fft, fftfreq

def fftrl(s, t, percent=0, n=0):
    """
    Forward Fourier transform for real-valued signal

    Forward fourier transform of a real-valued signal. 
    Unlike MATLAB's fft, this funtion returns only positive frequencies.
    For example, if the input signal has n samples, then there will be
    np.floor(n/2) + 1 frequency samples.

    This means that if n is an even number, then a time series of length
    n and one of length n+1 will produce frequency spectra of same length.
    However, only the first case will have a sample at the Nyquist frequency.

    Parameters:
    - s (ndarray): The input signal (trace or gather, i.e. vector or matrix)
    - t (ndarray): The time coordinate vector
    - percent (float): Specifies the length of a raised cosine taper to be 
                       applied to s (both ends) pripr to any padding.
                       Tapper is a percent of the length of s. Taper is applied
                       using MWINDOW. Default is 0%.
    - n: length to which the input trace is to be padded with zeros. Default is 
    the input length (no pad). 

    Returns:
    - spec: output spectrum
    - f: foutput requency sample vector
    """

    if n == 0:
        n = len(s)

    # Determine number of traces in ensemble
    [l, m] = s.shape
    ntraces = 1
    itr = 0 # transpose flag
    if l == 1:
        nsamps = m
        itr = 1
        s = s.T # transpose to column vector
    elif m == 1:
        nsamps = l
    else:
        nsamps = l
        ntraces = m

    if nsamps != len(t):
        t = t[0] + (t[1] - t[0]) * np.arange(nsamps)
        if len(t) != nsamps:
            raise ValueError('Time coordinate vector must have same length as trace')
    
    # Apply taper
    if percent > 0:
        mw = mwindow(nsamps, percent)
        mw = np.tile(mw, (ntraces, 1)).T
        s = s * mw
        del mw
    
    # Pad s if needed
    if nsamps < n:
        s = np.vstack((s, np.zeros((n - nsamps, ntraces))))
        nsamps = n

    # Transform the array
    spec = fft(s)
    spec = spec[:n // 2 + 1, :] # Save only positive frequencies
    del s

    # Build the frequency vector
    fnyq = 1. / (2 * (t[1] - t[0]))
    nf = spec.shape[0]
    df = 2 * fnyq / n
    f = np.linspace(0, fnyq, nf)

    if itr:
        f = f
        spec = spec.T

    return spec, f 


def mwindow(n, percent):
    """
    Creates an mwindow (boxcar with raised cosine tapers) of length n.

    MWINDOW returns the N-point Margrave window in a column vector. This 
    windows is a boxcar over the central samples (100-2*percent)*n/100 in 
    number, while it has a raised cosine (hanning style) taper on each end.
    If n is a vector, it is the same as mwindow(length(n)).

    Parameters:
    - n (int): The length of the window.
    - percent (float): The percentage of the window that is tapered. (default=10)

    Returns:
    w: The window vector of length n.
    """

    if percent < 0 or percent > 50:
        raise ValueError('Percent must be between 0 and 50')
    
    m = 2 * percent * n / 100
    m = 2 * np.floor(m / 2)
    h = np.hanning(int(m))
    w = np.concatenate([h[:int(m/2)], np.ones(n - int(m)), h[int(m/2):]])

    

