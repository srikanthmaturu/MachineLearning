# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import numpy as np
from timeit import default_timer as timer
from numba import cuda
from numba import vectorize
from final_project import *


#def VectorAdd(a, b, c):
#    for i in range(a.size):
#        c[i]=a[i]+b[i]

@vectorize(["float32(float32, float32)"], target='parallel')
def VectorAdd(a, b):
        return a * b

def main():
    N = 32000000
    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)
	
    start = timer()
    C = VectorAdd(A, B)
    vectoradd_time = timer() - start
    print("C[:5] = " + str(C[:5]))
    print("C[-5:] = " + str(C[-5:]))
	
    print("VectorAdd took %f seconds " % vectoradd_time)
	
if __name__ == '__main__':
	main()