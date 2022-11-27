
from svd import SVD
import numpy as np
from numpy.testing import *
import pandas as pd
import gc
import time
# from base import Mathutil, eighk


class TestSVD():
    # reading data from csv and stroing in 'data'
    df = pd.read_csv('ratings.csv', header=None)
    df.columns = ['userid','movieid','ratings','timestamp']
    df = df[1:];
    df = df.iloc[:,[0,1,2]]
    df = df.pivot(index='userid', columns='movieid', values='ratings');
    df = df.fillna(0.0);
    data = np.array(df, dtype='float')
    data = data[:]
    del df
    gc.collect()


    def test_compute(self):
        mdl = SVD(self.data)
        
        startt = time.clock()
        mdl.factorize()
        endt = time.clock()
        print('time', endt-startt)
    
        print (mdl.U.shape,mdl.U ,"\n" )
        print (mdl.S.shape,mdl.S ,"\n")
        print (mdl.V.shape,mdl.V ,"\n")

        # print the forbinium norm
        print(mdl.frob_norm_svd())

# main control
if __name__ == '__main__':
    ob = TestSVD()
    ob.test_compute()

