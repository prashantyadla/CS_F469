
import numpy as np
import scipy.sparse
from numpy.linalg import eigh
from scipy.misc import factorial

_EPS = np.finfo(float).eps

# return non-zero eigen value and coressponding eigen vector
def eighk(M, k=0):
    values, vectors = eigh(M)            
              
    # get rid of too low eigenvalues
    s = np.where(values > _EPS)[0]
    vectors = vectors[:, s] 
    values = values[s]                            
             
    # sort eigenvectors according to largest value
    idx = np.argsort(values)[::-1]
    values = values[idx]
    vectors = vectors[:,idx]

    # select only the top k eigenvectors
    if k > 0:
        values = values[:k]
        vectors = vectors[:,:k]

    return values, vectors


class Matutil():    
    _EPS = _EPS
    
    def __init__(self, data, k=-1, rrank=0, crank=0):
        self.data = data
        (self._rows, self._cols) = self.data.shape

        self._rrank = self._rows
        if rrank > 0:
            self._rrank = rrank
            
        self._crank = self._cols
        if crank > 0:            
            self._crank = crank
        
        self._k = k
    
    def frob_norm_svd(self):
        # frobenius norm: F = ||data - USV||   
        err = self.data[:,:] - np.dot(np.dot(self.U, self.S), self.V)
        err = np.sqrt(np.sum(err**2))
                            
        return err

    def frob_norm_cur(self):  
        # frobenius norm: F = ||data - CUR||   
        err = self.data[:,:] - np.dot(np.dot(self.U, self.S), self.V)
        err = np.sqrt(np.sum(err**2))
                            
        return err
    


