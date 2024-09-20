import numpy as np
from scipy.signal import hilbert, chirp
from scipy.fft import ifft
from fftrl import fftrl
from ifftrl import ifftrl

def padpow2(trin, flag=0):
    """
    Padpow pads the input trace to the next power of 2.
    
    flag = 1 --> If the trace is already an exact power of two, then 
                 its length will be doubled.
    flag = 0 --> If the trace is already an exact power of two, then 
                 its length will not be changed.
    
    trin : input trace
    """
    # Compute the next power of 2 for the length of trin
    n = 2 ** np.ceil(np.log2(len(trin))).astype(int)

    if n == len(trin) and flag == 1:
        n = 2 * np.ceil(np.log2(len(trin) + 1)).astype(int)
    
    # Check if trin is a row vector or column vector
    trin = np.array(trin)
    nr, nc = trin.shape if trin.ndim > 1 else (1, len(trin))

    if nr == 1:
        trout = np.hstack((trin, np.zeros(n - nc))) # row vector
    else:
        trout = np.vstack((trin, np.zeros((n - nr, nc)))) # column vector
    
    return trout

def filtf(trin, t, fmin=0, fmax=0, phase=0, max_atten=80):
    """
    Apply a bandass filter to a trace.

    Filters the input trace in the frequency domain.
    Trin is automatically padded to the next larger power of 2 
    and the pad is removed when passing trout to the output.

    Filter slopes are formed from Gaussian functions.

    trin : input trace
    t : time coordinate for trin
    fmin : a two element vector specifying:
        fmin[0] : 3db down point of filter on low end (Hz)
        fmin[1] : gaussian width on low end
    (default is [0, 5], set to [0, 0] for a low pass filter)

    fmax : a two element vector specifying:
        fmax[0] : 3db down point of filter on high end (Hz)
        fmax[1] : gaussian width on high end
    (default is [0, %10 of fnyquist], set to [0, 0] for a high pass filter)
    
    phase : 0 for zero phase, 1 for minimum phase
    (minimun phase filters are approximate in the sense that the output from
    this function is truncated to be the same length as the input. This works 
    fine as long as the trace being filtered is long compared to the impulse 
    response of the filter.
    Be wary of narrow band minimum phases filtered on short time series. The 
    result may not be minimum phase.)

    max_atten : maximum attenuation of filter in db
    trout : output trace
    """

    NT = int(np.ceil(0.1 * len(t)))
    trin = np.concatenate([np.zeros(NT), trin, np.zeros(NT)])
    trin = trin.reshape(-1, 1)
    t = (t[1] - t[0]) * np.arange(0, 2 * NT + len(t))

    # Set default

    if len(fmin) == 1:
        fmin = [fmin, 5]

    if len(fmax) == 1:
        fmax = [fmax, 0.1 * 1 / 2. * (t[1] - t[0])]

    # Convert to column vector 
    
    rr = trin.shape[0]
    cc = trin.shape[1]
    trflag = 0
    nt = len(t)
    if rr != nt and cc == nt:
        trin = trin.T
        trflag = 1
    elif rr != nt and cc != nt:
        print('time vector length not found in input matrix dimension, filtering columns.')
        return
    
    dbd = 3.0 # this controls the dbdown values of fmin and fmax
    nt = trin.shape[0]
    trinDC = np.ones((nt, 1)) * sum(trin) / nt
    trin = trin - trinDC

    # Forwward transform the trace
    trin = padpow2(trin)
    t = (t[1] - t[0]) * np.arange(0, nt)
    [Trin, f] = fftrl(trin, t)
    f = f.reshape(-1, 1)
    nf = len(f)
    df = f[1] - f[0]

    # Design low end gaussian
    if fmin[0] > 0:
        fnotl = fmin[0] + np.sqrt(np.log(10) * dbd/20) * fmin[1]
        gnot = 10 ** (-max_atten / 20.)
        glow = gnot + np.exp(-((f - fnotl) / fmin[1]) ** 2)

    if phase != 1:
        glow[0] = 0
    else:
        glow = 0
        fnotl = 0

    # Design high end gaussian
    if fmax[0] > 0:
        fnoth = fmax[0] - np.sqrt(np.log(10) * dbd/20.) * fmax[1]
        gnot = 10 ** (-max_atten / 20.)
        ghigh = gnot + np.exp(-((f - fnoth) / fmax[1]) ** 2)
    else:
        ghigh = 0
        fnoth = 0
    
    # Make filter
    fltr = np.ones(f.shape)
    nl = np.floor(fnotl / df).astype(int)[0] # Can fix one line from df[]
    nh = np.ceil(fnoth / df).astype(int)[0]

    if nl == 0:
        # If nl is 0, concatenate the first part of fltr with the high pass filter
        fltr = np.concatenate((fltr[:nh], ghigh[nh:]))
    
    elif nh == 0:
        # If nh is 0, concatenate the low pass filter with the second part of fltr
        fltr = np.concatenate((glow[:nl+1], fltr[nl+2:]))

    else:
        # General case: combine glow, fltr, and ghigh, with the specific indices
        fltr = np.concatenate([glow[:nl+1], fltr[nl+1:nf]]) * np.concatenate([fltr[:nh], ghigh[nh+0:len(f)]])
    
        # Normalize by the maximum absolute value
        fltr = fltr / max(abs(fltr))

    # Make min phase if required 
    if phase == 1:
        L1 = np.arange(len(fltr))
        L2 = np.arange(-len(fltr) - 1, 1, -1)

        symspec = np.concatenate((fltr[L1, np.conj(fltr[L2])]))
        cmpxspec = np.log(symspec) + 1j * np.zeros_like(symspec)

        fltr = np.exp(np.conj(hilbert(cmpxspec)))
    
    # Apply filter
    trout = ifftrl(Trin * (fltr[:len(f)] * np.ones((1, Trin.shape[1]))), f)[0]
    trout = trout[:nt, :]

    trinDC = trinDC * fltr[0]
    troutDC = np.ones((nt, 1)) * sum(trout) / nt

    trout = trout - troutDC + trinDC
    if trflag:
        trout = trout.T


    trout = np.delete(trout, slice(0, NT))
    trout = np.delete(trout, slice(-NT, None))

    return trout