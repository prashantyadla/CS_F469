import nltk
import os
import math
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import time 
from tkinter import *

PWD = "C:\\Users\\Prashant\\AppData\\Roaming\\nltk_data\\corpora\\movie_reviews\\neg\\"  # PATH OF THE CORPUS

class MakeDoc:

	def __init__(self):
		self.document_count=0
		self.term_frequency=0
		self.id_list=[]
		self.doc_term_count={}
		self.tfid={}

	def add_document_id(self,document_id,term_count):
		self.id_list.append(document_id)
		self.document_count=self.document_count+1;
		self.term_frequency+=term_count
		# tf of the term
		self.doc_term_count[document_id]=term_count
		# total frequency of the term 

	def calculate_tfid(self,N):
		for d_id in self.id_list:
			self.tfid[d_id] = (1+math.log(self.doc_term_count[d_id]))*(1+math.log(N/self.document_count))
			# calculating the tf-idf score of the term using (1+log(tf))*(1+log(N/df))



## ---------------------------------PHASE 1---------------------------
documents=[]
# corpus list
stop_words=stopwords.words("english")


start_time = time.time()

for file in [doc for doc in os.listdir(PWD) if doc.endswith(".txt")]:
	documents.append(file)

no_of_documents=len(documents)
# print (no_of_documents)
# print (documents)
terms={}
# vocabulary of the corpus 
tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer("english")


unstemmed_tokens = []

for doc_name in documents:
	file_t=open(PWD+doc_name,'r')
	text=file_t.read()
	tokenized_list = tokenizer.tokenize(text)
	# tokenizing the text and storing it in a list 
	unstemmed_tokens += tokenized_list

	stemmed_list = [stemmer.stem(words) for words in tokenized_list if words not in stop_words]
	# the tokenized list is stemmed 
	copy_tokenized_list=list(stemmed_list)

	for j in copy_tokenized_list:
		# calculating the inverted index using a list of dictionary 
		if (j in terms):
			if (doc_name in terms[j].id_list):
				continue;
			else:
				terms[j].add_document_id(doc_name,copy_tokenized_list.count(j))
		else:
			terms[j]=MakeDoc()
			terms[j].add_document_id(doc_name,copy_tokenized_list.count(j))

unstemmed_tokens = list(set(unstemmed_tokens)-set(stop_words))


key= terms.keys()
#for p in key:
#	print p, terms[p].document_count, terms[p].term_frequency

for p in key:
	# calculating the tf-idf score of the terms/stemmed vocabulary
	terms[p].calculate_tfid(no_of_documents)


print("Indexing done@ ",time.time()-start_time )
# indexing time for the corpus 
    
##------------------------------PHASE-2--------------------------------------------##


doc_normalization={}

# CALCULATING NORMALIZED LENGTH FOR EACH DOCUMENT
for doc_name in documents:
	value=0
	for j in terms:
		if (doc_name in terms[j].id_list):
			value=value+((terms[j].tfid[doc_name])*((terms[j].tfid[doc_name])))
	doc_normalization[doc_name] = math.sqrt(value)



while(True):
	print ("enter the query, enter q to quit")
	query1=input()
	# input query1 
	query1=query1.lower()
	if (len(query1)==1 and query1[0]=='q'):
		break
        
	tokenized_query=tokenizer.tokenize(query1)
	query_store=tokenized_query[:]
	tokenized_query=[word for word in tokenized_query if word not in stop_words]
	if (len(tokenized_query)==0):
		tokenized_query=query_store[:]
	query=[stemmer.stem(words) for words in tokenized_query]
	#print query

	q_dict = {}       # vector for query
	max_c = 0
	for y in query:
		if query.count(y) >= max_c:
			max_c = query.count(y)

	for w in query:
			if (w in terms):
				q_dict[w] = (1/max_c)*(1+query.count(w))*(1+math.log(no_of_documents/(terms[w].document_count)))

	#find out the cosine angle between the two vectors
	invalid_query_terms = []

	for x in tokenized_query:
		if (stemmer.stem(x)) not in key:
			invalid_query_terms.append(x)



        
	doc_score={}
	for q in query:
		if (q in terms):
			for j in terms[q].id_list:
				if (j in doc_score):
					doc_score[j] += (q_dict[q]*terms[q].tfid[j])/doc_normalization[j]
				else:
					doc_score[j] =(q_dict[q]*terms[q].tfid[j])/doc_normalization[j]

        # spell check function

	def spell_check(b):         # b is wrong query term 
		li = []
		for w in unstemmed_tokens:               
			if (b in w) and (abs(len(b)-len(w))<=3):
				li.append(w)
		return li
	if (invalid_query_terms):
		print("Suggestions for invalid query terms: ")

		for b in invalid_query_terms:
			if(spell_check(b)):
				print("Did u mean "+(", ".join(spell_check(b)))+ " instead of "+b +"?")

	# result holds the relevant documents

	result=doc_score.items()
	result=sorted(result,key=lambda x:x[1],reverse=True)
   	# sorting the result to pick the top 10 query 

	print ("The total no of results are ",len(result))
	if(len(result)>10):
		print("displaying top 10 results\n")
		c=10
		for r in result:
			c = c-1
			if c<0:
				break
			else:
				print (r[0])
	else:
		for r in result:
			print (r[0])
                
# PRINT r[1] for score

