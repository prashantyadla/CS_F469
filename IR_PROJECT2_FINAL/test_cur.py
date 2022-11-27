from cur import CUR
import numpy as np
from numpy.testing import *
from base import *
import pandas as pd
import gc
import time

class TestCUR():
    # #reading data from csv and stroing in 'data'
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
    # data = np.array([[1.0, 0.2, 1.0], [0.3, 1.0, 1.0],[0.1, 0.6, 0.4]])


    def test_compute(self):
        # mdl = CUR(self.data, rrank=2, crank=2)
        mdl = CUR(self.data, k=200, rrank=652, crank=90062)

        startt = time.clock()
        mdl.factorize()
        endt = time.clock()
        print('time', endt-startt)
        
        print (mdl.U.shape,mdl.U ,"\n" )
        print (mdl.S.shape,mdl.S ,"\n")
        print (mdl.V.shape,mdl.V ,"\n")

        # print the forbinium norm    
        print(mdl.frob_norm_cur())

        
if __name__ == '__main__':
    ob = TestCUR()
    ob.test_compute()