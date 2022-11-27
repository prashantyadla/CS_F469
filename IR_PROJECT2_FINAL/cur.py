
import numpy as np
import numpy
import scipy.sparse
from math import sqrt

# user defined pinv
from svd import pinv
from base import Matutil


class CUR(Matutil):    
    def __init__(self, data, k=-1, rrank=0, crank=0):
        Matutil.__init__(self, data, k=k, rrank=rrank, crank=rrank)
        
        self._rset = range(self._rows)
        self._cset = range(self._cols) 

    # returning the index of the samplec columns and row    
    def sample(self, s, probs):        
        prob_rows = np.cumsum( probs.flatten())            
        temp_ind = np.zeros(s, np.int32)
        np.random.seed(0)
        for i in range(s):            
            v = np.random.rand()
                        
            try:
                tempI = np.where(prob_rows >= v)[0]
                temp_ind[i] = tempI[0]        
            except:
                temp_ind[i] = len(prob_rows)
            
        return np.sort(temp_ind)
    
    # calculating the probility of the row and columns 
    def sample_probability(self):
        dsquare = self.data[:,:]**2    

        prow = np.array(dsquare.sum(axis=1), np.float64)
        pcol = np.array(dsquare.sum(axis=0), np.float64)
        
        prow /= prow.sum()
        pcol /= pcol.sum()   
        
        return (prow.reshape(-1,1), pcol.reshape(-1,1))
    
    # calculating the CUR 
    def computeUCR(self):
        # sampling value
        k = self._k
        c_no = k
        r_no = k

        np.random.seed(0)
        # id_c = np.random.randint(9060, size=c_no)
        # id_r = np.random.randint(650, size=r_no) 

        id_c = np.random.randint(self._rrank, size=c_no)
        id_r = np.random.randint(self._crank, size=r_no) 

        # np.random.seed(0)
        dsquare = self.data[:,:]**2 
        dsquare = np.sum(dsquare)

        
        # calculating C and R
        self._C = self.data[:, id_c]/sqrt(k*dsquare)
        self._R = self.data[id_r, :]/sqrt(k*dsquare)

        # calculating the W, the intersection of C and R
        self.W = np.zeros((r_no, c_no))
        for i in range(c_no):
            for j in range(r_no):
                self.W[i][j] = self.data[ id_r[i] ][ id_c[j] ]/sqrt(k*dsquare)

        # pseudo inverse of the W, user defined function
        self._U = pinv(self.W, k)
        
        # set some standard (with respect to SVD) variable names 
        self.U = self._C
        self.S = self._U
        self.V = self._R

    # factorisation od the given matrix 
    def factorize(self):      
                                    
        self.computeUCR()



