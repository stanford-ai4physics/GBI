import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def gen_sig(amplitude=100.0):
    """Generate toy signal data mimicking gravitational wave detection.
    
    Parameters
    ----------
    amplitude : float, default=100.0
        Signal amplitude scaling factor
    
    Returns
    -------
    numpy.ndarray
        Generated signal data with gravitational wave-like characteristics
    """
    T = 20.0 # ms
    delta = 5. # ms
    A = amplitude
    return gen_data(A,T,delta,0)

def gen_bg():
    """Generate toy background data for gravitational wave analysis.
    
    Returns
    -------
    numpy.ndarray
        Generated background data without signal components
    """
    T = 20.0 # ms
    delta = 5. # ms
    A = 0.0
    return gen_data(A,T,delta,0.25)

def gen_data(A, T, delta, noise):
    """Generate simulated gravitational wave detector data.
    
    Creates toy data mimicking gravitational wave signals in dual detectors
    with configurable amplitude, period, time delay, and noise levels.
    
    Parameters
    ----------
    A : float
        Signal amplitude
    T : float
        Signal period in milliseconds
    delta : float
        Time delay between detectors in milliseconds
    noise : float
        Gaussian noise standard deviation
        
    Returns
    -------
    numpy.ndarray, shape (100, 5)
        Generated data with columns [time, H_detector, L_detector, H+L, H-L]
        
    Notes
    -----
    This is a toy model for testing R-Anode methodology on simple
    time-series data before applying to particle physics datasets.
    """
    data = []
    for t in np.linspace(0,300,num=300): #ms
        h = A * np.sin(2*np.pi*t / T) * scipy.stats.norm.pdf(t,loc=150,scale=20) + np.random.normal(scale=noise)
        l = A * np.sin(2*np.pi*(t+delta) / T)* scipy.stats.norm.pdf(t,loc=150+delta,scale=20) + np.random.normal(scale=noise)
        data.append( [ t,h,l, h+l, h-l])

    return np.array(data)


sdata = gen_sig()
bdata = gen_bg()

plt.plot(sdata[:,0],sdata[:,1],label="H")
plt.plot(sdata[:,0],sdata[:,2],label="L")
plt.plot(sdata[:,0],sdata[:,1]+sdata[:,2],label="H+L",linestyle=":")
plt.plot(sdata[:,0],sdata[:,1]-sdata[:,2],label="H-L",linestyle=":")
plt.xlabel("Time [ms]")
plt.ylabel("Strain")
plt.legend()
plt.savefig("sigs.pdf")
plt.clf()

plt.plot(bdata[:,0],bdata[:,1],label="H")
plt.plot(bdata[:,0],bdata[:,2],label="L")
plt.plot(bdata[:,0],bdata[:,1]+bdata[:,1],label="H+L",linestyle=":")
plt.plot(bdata[:,0],bdata[:,1]-bdata[:,1],label="H-L",linestyle=":")
plt.legend()
plt.xlabel("Time [ms]")
plt.ylabel("Strain")
plt.savefig("bgs.pdf")
