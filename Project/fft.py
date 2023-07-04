import numpy
import math
from sys import argv 



def gen_dft(len):
    roots = []
    w = math.e ** ((math.pi * 2j)/ len)
    for i in range(len):
        roots.append(w**i)
    roots_unity = numpy.array(roots)
    dft = numpy.vander(roots_unity,increasing=True)
    return dft


def gen_diag(len):
    roots = []
    w = math.e ** ((math.pi * 2j)/ len)
    for i in range(int(len/2)):
        roots.append(w**i)
    Dmat = numpy.diag(numpy.array(roots))
    return Dmat


def combi_matrix(len):
    a = int(len/2)
    combi = numpy.block([
        [numpy.eye(a),gen_diag(len)],
        [numpy.eye(a),gen_diag(len)*-1]
    ])
    return combi


def gen_fft(len):
    if(len != 1):
        l = int(len/2)
        fft_recursive = numpy.block([
            [gen_fft(l),numpy.zeros((l,l))],
            [numpy.zeros((l,l)),gen_fft(l)]
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
        permutation_mat = numpy.array(even + odd)
        combi = combi_matrix(len)
        f = (combi_matrix(len) @ fft_recursive) @ permutation_mat
        return f
    
    else:
        return numpy.array([1])
        

##if(len(argv) != 2):
##    print("Wrong arguments")
##    print("fft.py <n for Fn>")
##
##else:
##    print("DFT:")
##    d = gen_dft(int(argv[1]))
##    print(numpy.round(d,3))
##
##    print("\n\nFFT:")
##    f = gen_fft(int(argv[1]))
##    print(numpy.round(f,3))

