from copy import deepcopy
import numpy as np
from scipy.sparse import coo_matrix

def mat_decompose(H:np.matrix):
    #Tensrosized reconstruction method: O(8^n)
    # Normal method: O(16^n)
    # mat = np.zeros(self.coef_matrix.shape) 
    # for p in self.poly:
    #   mat += p.coef*p.matrix
    #mat = self.coef_matrix
    n1, n2 = H.shape
    assert n1 == n2, "The given matrix must be a square matrix."
    n= int(np.log2(n1))
    l = n1
    for i in range(n):
        m = int(2**i) # Number of submatrix
        l = int(l/2) # Sub matrix size, square
        for j in range(m):
            for k in range(m):
                num_i = j*(2*l) # Initial position of sub matrix row
                num_j = k*(2*l) # Initial position of sub matrix column
                
                # There is no problem of inplace operators.
                
                # I-Z
                H[num_i: num_i+l, num_j:num_j+l]        += H[num_i+l: num_i+2*l, num_j+l:num_j+2*l] 
                H[num_i+l: num_i+2*l, num_j+l:num_j+2*l] = H[num_i: num_i+l, num_j:num_j+l] - 2*H[num_i+l: num_i+2*l, num_j+l:num_j+2*l]
                # X-Y
                H[num_i: num_i+l, num_j+l:num_j+2*l] +=  H[num_i+l: num_i+2*l, num_j:num_j+l] 
                H[num_i+l: num_i+2*l, num_j:num_j+l]  =  H[num_i: num_i+l, num_j+l:num_j+2*l] - 2*H[num_i+l: num_i+2*l, num_j:num_j+l]
                H[num_i+l: num_i+2*l, num_j:num_j+l] *= 1j

    H *= (1/(2**n))
    return H

def h_decompose(H:np.matrix):
    h = deepcopy(H)
    return coo_matrix(mat_decompose(h))