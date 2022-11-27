import numpy
import random
import math
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
import time



t = []




# function to sample k arrays based on the probabilty distribution
def choose_rand(probarray):
	"""
	return random columns or row depending on its probability
	"""
	sortarr=sorted(probarray.iteritems(),key=lambda (k,v):(v,k))
	index=[]
	prob=[]
	elements=0
	for i,val in sortarr:
		elements+=1
		index.append(i)
		prob.append(val)
	#print index
	#print prob	
	for i in range(1,elements):
		prob[i]=prob[i]+prob[i-1]
	x=random.random()
	#print prob
	i=0;
	#print x
	while(x>prob[i]):
		i+=1
	return index[i]


def calculate_frobenius_norm(orig,result):
	"""
	returns the frobenius error given the original matrix and the newly constructed matrix
	"""
	[no_row,no_col]=orig.shape
	error=0;
	for i in range(0,no_row):
		for j in range(0,no_col):
			error+=(orig[i,j]-result[i,j])**2
	return math.sqrt(error)		
			

# function to calculate CUR decomposition
def CURCalculate(A,samplerow,samplecol):
	#A=numpy.matrix('4,0,0,1;2,1,0,0;0,0,0,3')
	#print(A)
	n_row,n_col=A.shape                             # actual number of rows and columns
	n_sample_row=samplerow                          # storing sample number of rows
	n_sample_column=samplecol                       # storing sample number of columns
	Acalc=numpy.zeros(shape=(n_row,n_col))          # filling result matrix with initial 0
	Cmat=numpy.zeros(shape=(n_row,n_sample_column)) # C =0 
	Rmat=numpy.zeros(shape=(n_sample_row,n_col))    # R = 0
	PCOL={}
	PROW={}
	sampled_columns=[]
	sampled_rows=[]
	sum_square=0
	for t in A.getA1():                             # calculating sum of squares of each element
		sum_square+=(t**2)                        
	#print ("sum of square of elements is "+str(sum_square))
	for i in range(0,n_col):                # calculating sum of squares for each column
	  COL=	A.getT()[i].getA1()
	  add=0
	  for x in COL:
		add+=x**2
	  PCOL[i]= float(add)/sum_square         # storing probability values for each column
	#Construct C
	for i in range(0,n_sample_column):
		j=choose_rand(PCOL)                  # sampling columns based on probability
		#print j
		sampled_columns.append(j)            # storing the sampled columns
		for k in range(0,n_row):
			Cmat[k,i]=float(A[k,j])/math.sqrt(n_sample_column*PCOL[j])       # storing the sampled columns

	for i in range(0,n_row):
		ROW=A[i].getA1()
		add=0
		for x in ROW:
			#print x
			add+=x**2
		PROW[i]= float(add)/sum_square		# storing probability values for each row
	#Make R
	for i in range(0,n_sample_row):
		j=choose_rand(PROW)					# sampling rows based on probability
		sampled_rows.append(j)				# storing the sampled rows
		for k in range(0,n_col):
			Rmat[i,k]=A[j,k]/math.sqrt(n_sample_row*PROW[j])			# storing the sampled rows
	print ("\nC matrix is :-\n")
	print Cmat
	print ("\nR matrix is:-\n")
	print Rmat	

	#Construct U

	Wmat=numpy.zeros(shape=[n_sample_row,n_sample_column])
	Umat=numpy.zeros(shape=[n_sample_row,n_sample_column])
	#print ("\nSampled Rows are:\n")
	#print(sampled_rows)
	#print ("\nSampled Columns are:\n")
	#print(sampled_columns)
	for i in range(0,n_sample_row):
		for j in range(0,n_sample_column):
			Wmat[i,j]=A[sampled_rows[i],sampled_columns[j]]
	#print "\nOriginal Matrix A is:\n"
	#print(A)
	#print "\n W matrix is :\n"
	#print(Wmat)
	Umat=numpy.linalg.pinv(Wmat)
	print("\nU matrix is :-\n")
	print(Umat)
	Acalc=numpy.dot(Cmat,numpy.dot(Umat,Rmat))
	#print '\nFrobenius Error\n'
	frob_error=calculate_frobenius_norm(A,Acalc)
	return frob_error,Acalc,Cmat,Umat,Rmat




if __name__ == '__main__':
	random.seed(1234567);
	#A=numpy.matrix('4,0,0,1;2,1,0,0;0,0,0,3')
	#A = numpy.matrix('1,1,1,0,0;3,3,3,0,0;4,4,4,0,0;5,5,5,0,0;0,0,0,4,4;0,0,0,5,5;0,0,0,2,2')

	# uncomment below code for testing any dataset
	# and comment the above code
	
	df = pd.read_csv('ratings.csv', header=None)
	df.columns = ['userid','movieid','ratings','timestamp']
	df = df[1:]
	df = df.iloc[:,[0,1,2]]
	df = df.pivot(index='userid', columns='movieid', values='ratings');
	df = df.fillna(0.0);
	A = np.array(df, dtype='float')
	A = A[:]
	del df
	gc.collect()
	
	start = time.clock()
	#A=numpy.random.random((10,10))
	A=numpy.asmatrix(A)
	sum_error=0
	for i in range(0,10):
		err,Acalc,Cmat,Umat,Rmat=CURCalculate(A,150,250)               
		print err
		t.append(err)
		sum_error+=err
	end = time.clock()
	o = np.arange(0.,10.,1)

	plt.ylabel('frobenius norm error')
	plt.xlabel('example considered')
	plt.axis([0,10,20,8000])
	plt.plot(o,t,'r--')
	plt.show()


	print 'Average Frobenius Error for matrix :'+str(float(sum_error)/10)	
	print((end-start))



