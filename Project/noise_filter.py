import numpy as np
import matplotlib.pyplot as plt
import math
##plt.rcParams['figure.figsize'] = [16,12]
plt.rcParams.update({'font.size': 18})

##def DFT_slow(x):
##    """Compute the discrete Fourier Transform of the 1D array x"""
##    x = np.asarray(x, dtype=float)
##    N = x.shape[0]
##    n = np.arange(N)
##    k = n.reshape((N, 1))
##    M = np.exp(-2j * np.pi * k * n / N)
##    return np.dot(M, x)
##
##def FFT(x):
##    """A recursive implementation of the 1D Cooley-Tukey FFT"""
##    x = np.asarray(x, dtype=float)
##    N = x.shape[0]
##    if N % 2 > 0:
##        raise ValueError("size of x must be a power of 2")
##    elif N <= 32:  # this cutoff should be optimized
##        return DFT_slow(x)
##    else:
##        X_even = FFT(x[::2])
##        X_odd = FFT(x[1::2])
##        factor = np.exp(-2j * np.pi * np.arange(N) / N)
##        return np.concatenate([X_even + factor[:int(N / 2)] * X_odd,
##                               X_even + factor[int(N / 2):] * X_odd])


def gen_dft(len):
    roots = []
    w = math.e ** ((math.pi * 2j)/ len)
    for i in range(len):
        roots.append(w**i)
    roots_unity = np.array(roots)
    dft = np.vander(roots_unity,increasing=True)
    return dft


def gen_diag(len):
    roots = []
    w = math.e ** ((math.pi * 2j)/ len)
    for i in range(int(len/2)):
        roots.append(w**i)
    Dmat = np.diag(np.array(roots))
    return Dmat


def combi_matrix(len):
    a = int(len/2)
    combi = np.block([
        [np.eye(a),gen_diag(len)],
        [np.eye(a),gen_diag(len)*-1]
    ])
    return combi


def gen_fft(len):
    if(len != 1):
        l = int(len/2)
        fft_recursive = np.block([
            [gen_fft(l),np.zeros((l,l))],
            [np.zeros((l,l)),gen_fft(l)]
        ])
        b = []
        for i in range(len):
            k = []
            for j in range(len):
                if (j==i):
                    k.append(1)
                else:
                    k.append(0)
            b.append(k)
        
        even = [b[i] for i in range(len) if (i % 2 == 0)] 
        odd = [b[i] for i in range(len) if (i % 2 != 0)] 
        permutation_mat = np.array(even + odd)
        combi = combi_matrix(len)
        f = (combi_matrix(len) @ fft_recursive) @ permutation_mat
        return f
    
    else:
        return np.array([1])
        
    

dt = 1/1024
t = np.arange(0, 1, dt)
##f = np.sin(2*np.pi*310*t) + np.sin(2*np.pi*120*t)
f = np.sin(2*np.pi*120*t)
f_clean = f
f = f + 2.5*np.random.randn(len(t))

plt.plot(t, f, color = 'c', linewidth = 1.5, label = 'Noisy')
plt.plot(t, f_clean, color = 'k', linewidth = 2, label = 'Clean')
plt.xlim(t[0], t[-1])
plt.legend()
##plt.show()


n = len(t)
Fn = gen_fft(n)
##fhat = np.fft.fft(f,n)
##fhat = FFT(f)
fhat = np.matmul(Fn, f)
PSD = fhat * np.conj(fhat)/n
freq = (1/(dt*n)) * np.arange(n)
L = np.arange(1, np.floor(n/2), dtype = 'int')

fig, axs = plt.subplots(2,1)

plt.sca(axs[0])

plt.plot(t, f, color = 'c', linewidth = 1.5, label = 'Noisy')
plt.plot(t, f_clean, color = 'k', linewidth = 2, label = 'Clean')
plt.xlim(t[0], t[-1])
plt.legend()

plt.sca(axs[1])
plt.plot(freq[L], PSD[L], color = 'c', linewidth = 2, label = 'Peak Spectrum')
plt.xlim(freq[L[0]], freq[L[-1]])
plt.legend()
plt.show()


indices = PSD > 100
PSDclean = PSD * indices
fhat = indices * fhat
ffilt = np.fft.ifft(fhat)

fig, axs = plt.subplots(3,1)

plt.sca(axs[0])

plt.plot(t, f, color = 'c', linewidth = 1.5, label = 'Noisy')
plt.plot(t, f_clean, color = 'k', linewidth = 2, label = 'Clean')
plt.xlim(t[0], t[-1])
plt.legend()

plt.sca(axs[1])
plt.plot(t, ffilt, color = 'k', linewidth = 2, label = 'Filtered')
plt.xlim(t[0], t[-1])
plt.legend()


plt.sca(axs[2])
plt.plot(freq[L], PSD[L], color = 'c', linewidth = 2, label = 'Noisy')
plt.plot(freq[L], PSDclean[L], color = 'k', linewidth = 1.5, label = 'Filtered')
plt.xlim(freq[L[0]], freq[L[-1]])
plt.legend()
##plt.show()


