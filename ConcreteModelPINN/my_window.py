import numpy as np
from scipy.signal.windows import tukey

def my_window(data, dt, t1, t2, dist, Vp):
    """
    Apply a windowing function to seismic data.

    Parameters:
    - data (ndarray): The seismic data array of shape (nt, Nr).
    - dt (float): The time sampling interval.
    - t1 (float): The start time of the window.
    - t2 (float): The end time of the window.
    - dist (ndarray): The distance array of shape (Nr,).
    - Vp (float): The velocity of the seismic wave.

    Returns:
    - out (ndarray): The windowed seismic data array of shape (nt, Nr).
    """
    nt = data.shape[0]
    Nr = data.shape[1]

    out = np.zeros((nt, Nr))
    N1 = np.ceil(t1 / dt).astype(int)
    N2 = np.ceil(t2 / dt).astype(int)
    w1 = tukey(2 * N1, alpha=0.5)
    w2 = tukey(2 * N2, alpha=0.5)

    # References
    ref = np.round(dist / Vp / dt).astype(int)

    for j in range(Nr):
        b = ref[j]
        start = b - N1
        stop = b + N2
        
        if start < 1:
            start = 1
        if stop > nt:
            stop = nt

        w = np.zeros(nt)
        w[start:b+1] = w1[N1 - (b - start):N1+1]
        
        w[b + 1: stop + 1] = w2[:stop - b]
        out[:, j] = w * data[:, j]

        if dist[j] > 0:
            out[:, j] /= np.sqrt(np.sum(out[:, j] ** 2))
        else:
            out[:, j] = 0

    return out