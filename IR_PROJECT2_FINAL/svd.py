
from numpy.linalg import eigh
import time
import scipy.sparse
import numpy as np
from base import Matutil, eighk


# pinv() : Compute the pseudoinverse of a Matrix for CUR
def pinv(A, k=-1, eps= np.finfo(float).eps):    
    # Compute Pseudoinverse of a matrix   
    svd_mdl =  SVD(A, k=k)
    svd_mdl.factorize()
    
    S = svd_mdl.S
    Sdiag = S.diagonal()
    Sdiag = np.where(Sdiag>eps, 1.0/Sdiag, 0.0)
    
    for i in range(S.shape[0]):
        S[i,i] = Sdiag[i]

    if scipy.sparse.issparse(A):            
        A_p = svd_mdl.V.transpose() * (S * svd_mdl.U.transpose())
    else:    
        A_p = np.dot(svd_mdl.V.T, np.core.multiply(np.diag(S)[:,np.newaxis], svd_mdl.U.T))

    return A_p


class SVD(Matutil):    

    def _compute_S(self, values):
        self.S = np.diag(np.sqrt(values))
        # and the inverse of it
        S_inv = np.diag(np.sqrt(values)**-1.0)
        return S_inv

   
    def factorize(self):    
        def _right_svd():            
            AA = np.dot(self.data[:,:], self.data[:,:].T)
            # argsort sorts in ascending order -> access is backwards
            values, self.U = eighk(AA, k=self._k)

            # compute S
            self.S = np.diag(np.sqrt(values))
            
            # inverse of it
            S_inv = self._compute_S(values)
                    
            # compute V 
            self.V = np.dot(S_inv, np.dot(self.U[:,:].T, self.data[:,:]))    
            
        
        def _left_svd():
            AA = np.dot(self.data[:,:].T, self.data[:,:])
            
            values, Vtmp = eighk(AA, k=self._k)
            self.V = Vtmp.T 

            # inverse of it
            S_inv = self._compute_S(values)

            # conpute U
            self.U = np.dot(np.dot(self.data[:,:], self.V.T), S_inv)                


        if self._rows >= self._cols:          
            _left_svd()
        else:       
            _right_svd()

