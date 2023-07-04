import math
import numpy as np
import time
import matplotlib.pyplot as plt


### Discrete Fourier Transform Implementation
### Time complexity O(n^2)

def gen_dft(len):
    roots = []
    w = math.e ** ((math.pi * 2j)/ len)
    for i in range(len):
        roots.append(w**i)
    roots_unity = np.array(roots)
    dft = np.vander(roots_unity,increasing=True)
    return dft

### Fast Fourier Transform Implementation
### Time complexity O(nlog_n)

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
        b = np.eye(len)
        b.tolist()    
        even = [b[i] for i in range(len) if (i % 2 == 0)] 
        odd = [b[i] for i in range(len) if (i % 2 != 0)] 
        permutation_mat = np.array(even + odd)
        combi = combi_matrix(len)
        f = (combi_matrix(len) @ fft_recursive) @ permutation_mat
        return f
    
    else:
        return np.array([1])


### Simulating Clean and Noisy Signals

dt = 1/1024 #1024 samples (2^10)
t = np.arange(0, 1, dt)
##f = np.sin(2*np.pi*150.8*t)    #sin wave of frequency 150.8hz
f = np.sin(2*np.pi*256*t) + np.sin(2*np.pi*123*t)    #sum of 2 frequencies
clean_signal = f
noisy_signal = f + 2.5*np.random.randn(len(t))

##plt.plot(t, noisy_signal, color = 'red', linewidth = 1, label = 'Noisy')
##plt.plot(t, clean_signal, color = 'green', linewidth = 1, label = 'Clean')
##plt.xlim(t[0], t[-1])
##plt.xlabel('Time')
##plt.ylabel('Amplitude')
##plt.title('Simulating Clean and Noisy Signals')
##plt.legend()
##plt.show()

### Apply FFT to compute component frequencies

n = len(t)

start = time.clock();

##FFT_matrix = gen_fft(n)
##noisy_freq = np.matmul(FFT_matrix, noisy_signal)    #frequency domain
##
DFT_matrix = gen_dft(n)
noisy_freq = np.matmul(DFT_matrix, noisy_signal)

##noisy_freq = np.fft.fft(noisy_signal)

end = time.clock()

PSD = noisy_freq * np.conj(noisy_freq)/n    #compute Power Spectral Density
##freq = (1/(dt*n)) * np.arange(n)
freq = np.arange(1/dt, dtype = 'int')

print((end - start)*1000)




##plt.plot(freq, PSD, color = 'blue', linewidth = 1, label = 'Peak Spectrum')
##plt.xlim(freq[0], freq[math.floor(n/2)])
##plt.xlabel('Frequency')
##plt.legend()
##plt.show()


### Filter Signal and Inverse FFT to produce Clean Signal

threshold = PSD > 100  ##Threshold
PSD_clean = PSD * threshold
clean_freq = threshold * noisy_freq
filtered_signal = np.fft.ifft(clean_freq)


fig, axs = plt.subplots(2,1)

##plt.sca(axs[0])
##plt.plot(freq, PSD, color = 'red', linewidth = 1, label = 'Noisy')
##plt.plot(freq, PSD_clean, color = 'green', linewidth = 1, label = 'Filtered')
##plt.xlim(freq[0], freq[math.floor(n/2)])
##plt.legend()
##
##
##plt.sca(axs[1])
##plt.plot(t, filtered_signal, color = 'green', linewidth = 1, label = 'Filtered')
##plt.xlim(t[0], t[-1])
##plt.ylim(-5, 5)
##plt.legend()
##plt.show()

def time_complexity():
    n = 1024
    for i in range(6):
        start = time.clock()
        FFT_matrix = gen_fft(n)
        end = time.clock()
        print((end - start)*1000)
        n *= 2

