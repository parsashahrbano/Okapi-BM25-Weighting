# from ctypes.wintypes import WORD
# from tracemalloc import stop
import sys
import os
import spacy
import numpy as np
from numpy.linalg import norm
import math
import pandas as pd
import collections
from spacy.tokenizer import Tokenizer
from spacy import displacy
from spacy import tokens
from spacy.tokens import DocBin
import random
import re
nlp = spacy.load("en_core_web_sm")
from spacy.lang.en.stop_words import STOP_WORDS
from time import time
import nltk  
from sklearn.datasets import load_files  
# nltk.download('stopwords')  
import pickle  
# from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency


def get_sentens(doc, sent_number):
	d = nlp(doc)
	index=1
	for s in d.sents:
		if index == int(sent_number) + 1:
			return s
		else:
			index += 1

def bow_cosine(d1,sent_list):
	bow_list=[]
	list_one=[]
	list_two=[]
	vector_s1=[]
	vector_s2=[]
	# s1=nlp(d1)
	# for i in s1.sents:
	for tok in d1:
		if tok.lemma_.lower() not in list_one:
			list_one.append(tok.lemma_.lower())	
	# print(list_one)
	for j in sent_list:
		# m=nlp(j)
		# for k in m.sents:
		for n in j:
			if n.lemma_.lower() not in list_two:
				list_two.append(n.lemma_.lower())
	# print(list_two)
	for l in list_one:
		if l not in bow_list:
			bow_list.append(l)
	# print(bow_list)
	for o in list_two:
		if o not in bow_list:
			bow_list.append(o)
	# print(bow_list)
	#create vector
	for word in bow_list:
		if word in list_one:
			vector_s1.append(1)
		else:
			vector_s1.append(0)
	for wd in bow_list:
		if wd in list_two:
			vector_s2.append(1)
		else:
			vector_s2.append(0)
	# print(vector_s1)
	# print(vector_s2)
	#nominator
	cosine_numinator=0
	for m in range(0,len(vector_s1)):
		cosine_numinator+=vector_s1[m]*vector_s2[m]
	# print(cosine_numinator)
	#denominator
	power1_r1=0
	r1_denom=0
	for p in range(0,len(vector_s1)):
		power1_r1+=vector_s1[p]**2
		r1_denom=math.sqrt(power1_r1)
	# print(r1_denom)
	power2_r2=0
	r2_denom=0
	for q in range(0,len(vector_s2)):
		power2_r2+=vector_s2[q]**2
		r2_denom=math.sqrt(power2_r2)
	# print(r2_denom)
	cosine_denominator=r1_denom*r2_denom
	# print(cosine_denominator)
	#cosine_fraction
	if cosine_denominator != 0:
		cosine_fraction= cosine_numinator/float(cosine_denominator)
	else:
		cosine_fraction=0
	# print(cosine_fraction)
	return cosine_fraction


def jaccard_similarity(list1, list2):
	s1 = set(list1)
	s2 = set(list2)
	if len(s1.union(s2)):
		return float(len(s1.intersection(s2)) / len(s1.union(s2)))
	else:
		return 0



def okapi_bm25(s1,s2,k1=1.2,b=0.75):
	# calculate average document length
	avgdl = (len(s1) + len(s2)) / 2
	# calculate IDF for each term
	idfs = {}
	for term in set(s1 + s2):
		df = sum(1 for doc in [s1, s2] if term in doc)
		idf = math.log((len([s1, s2]) - df + 0.5) / (df + 0.5))
		idfs[term] = abs(idf)
	# calculate score for each sentence
	scores = []
	for i, terms in enumerate([s1, s2]):
		score = 0
		for term in terms:
			tf = terms.count(term)
			score += idfs[term] * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (len(terms) / avgdl)))
		scores.append(score)
	return(scores[1])




def cosine_similarity(s1, s2, d1, d2):
	tf_s1 = {}
	tf_s2 = {}
	for w in s1:
		if w in tf_s1:
			tf_s1[w] += 1
		else:
			tf_s1[w] = 1	
	for w in s2:
	    if w in tf_s2:
	        tf_s2[w] += 1
	    else:
		    tf_s2[w] = 1
	tf_idf_s1 = {}
	tf_idf_s2 = {}
	for w in tf_s2:
		tf_idf_s2[w] = tf_s2[w]/float(d2[w])
	for w in tf_s1:
		tf_idf_s1[w] = tf_s1[w]/float(d1[w])
	zigma = 0
	for i in tf_s1:
		if i in tf_s2:
			zigma += tf_idf_s1[i]*tf_idf_s2[i]
	r1 = 0
	r2 = 0
	for w in tf_s1:
		r1 += (tf_s1[w]/float(d1[w])) ** 2
	for w in tf_s2:
		r2 += (tf_s2[w]/float(d2[w])) ** 2
	if r1 != 0 and r2 !=0:
		cosine_sim = zigma/float(math.sqrt(r1) * math.sqrt(r2))
	else:
		cosine_sim = 0
	return cosine_sim
def identitysimilarity(s1, s2):
	tfs1={}
	tfs2={}
	for i in s1:
		if i in tfs1:
			tfs1[i]+=1
		else:
			tfs1[i]=1
	for i in s2:
		if i in tfs2:
			tfs2[i]+=1
		else:
			tfs2[i]=1
	l_s1=len(s1)
	l_s2=len(s2)
	absolute_value=abs(l_s1-l_s2)
	idf={}
	for k in tfs1:
		if k in tfs2:
			idf[k]=tfs1[k]+tfs2[k]
		else:
			idf[k]=tfs1[k]
	for k in tfs2:
		if k not in idf:
			idf[k]=tfs2[k]
	zigma=0
	for i in tfs1:
		if i in tfs2:
			zigma+=idf[i]+absolute_value
	if zigma==0:
		fraction_1=0
	else:
		fraction_1=1/float(zigma)
	sec_fraction=0
	for i in tfs1:
		if i in tfs2:
			sec_fraction+=idf[i]/float(1+abs(tfs1[i]-tfs2[i]))
	prod_ident=fraction_1*sec_fraction
	return prod_ident

def similarity(doc1, doc2, topics):
	doc_bin = DocBin(attrs=["LEMMA", "ENT_IOB", "ENT_TYPE"], store_user_data=True)
	d1 = nlp(doc1)
	d2 = nlp(doc2)
	idf_d1 = {}
	idf_d2 = {}
	norm_list=[]
	okapi_score = []
	cosine_score = []
	jaccard_score = []
	identity_score=[]
	topic_score = []
	word_str=""
	for tok in d1:
		if tok.is_stop == False and tok.text.isalpha() == True :
			t = tok.lemma_.lower()
			word_str+=t
			if t in idf_d1:
				idf_d1 [t] += 1
			else:
				idf_d1 [t] = 1
	for tok in d2:
		if tok.is_stop == False and tok.text.isalpha() == True :
			t = tok.lemma_.lower()
			if t in idf_d2:
				idf_d2 [t] += 1
			else:
				idf_d2 [t] = 1
	# print(idf_d1)
	jacard_time = 0.0
	identity_time = 0.0
	for sentence1 in d1.sents:
		score_list1 = []
		score_list2 = []
		score_list3 = []
		score_list4 = []
		score_list5 = []
		s1 = [token.lemma_.lower() for token in sentence1 if token.is_stop == False and token.text.isalpha() == True]
		for sentence2 in d2.sents:
			s2 = [token.lemma_.lower() for token in sentence2 if token.is_stop == False and token.text.isalpha() == True]
			score_list1.append(jaccard_similarity(s1, s2))
			score_list2.append(okapi_bm25(s1,s2,k1=1.2,b=0.75))
			score_list3.append(identitysimilarity(s1, s2))
			score_list4.append(topic_signature_similarity(topics, s1, s2, idf_d1, idf_d2))
			score_list5.append(cosine_similarity(s1, s2, idf_d1, idf_d2))
		jaccard_score.append(score_list1)
		sumu=sum(score_list2)
		l_t=[]
		for sc in score_list2:
			if sumu==0:
				l_t.append(0)
			else:
				frac=sc/sumu
				l_t.append(frac)
		okapi_score.append(l_t)
		identity_score.append(score_list3)
		topic_score.append(score_list4)
		cosine_score.append(score_list5)
		# print(okapi_score)
	return jaccard_score ,okapi_score, identity_score, topic_score , cosine_score

def hits (harmunic_dict):
	neighbors = {}
	hits = {}
	for k, d in harmunic_dict.items():
		doc1 = k[0:int(k.find("_"))]
		doc2 = k[int(k.find("_"))+1:len(k)+1]
		for index,data in enumerate(d):
			vertex = doc1 + '_S' + str(index)
			neighbor_list = []
			if vertex not in hits:
				hits[vertex] = 1
			else:
				neighbor_list = neighbors[vertex]
			for ind , data2 in enumerate(data):
				tmp = {}
				if data2 != 0 :
					tmp[doc2 + '_S' + str(ind)] = data2
					neighbor_list.append(tmp)
			neighbors[vertex] = neighbor_list
	replit = 10
	epsilon = 0.0000008
	new_hits = {}
	for r in range(replit):
		# print('hits repets : %d' % r , end='\r')
		convergence = 0
		for vi , hits_vi in hits.items():
			sum_vi = 0
			for n_vi in neighbors[vi]:
				vj = list(n_vi.keys())[0]
				dj = list(n_vi.values())[0]
				sum_vi += dj * hits[vj]
			new_hits[vi] = sum_vi
		#normalize and convergence
		norm = sum (new_hits.values())
		for vi , hits_vi in new_hits.items():
			new_hits[vi] = hits_vi / norm
			convergence += ( new_hits[vi] - hits[vi] ) **2 
		if convergence <= epsilon:
			break
		else:
			hits = new_hits.copy()
	# print()
	return new_hits

def pageRank (harmunic_dict):
	page_rank = {}
	neighbors = {}
	#initialaze PageRank and find neighbors
	for k, d in harmunic_dict.items():
		doc1 = k[0:int(k.find("_"))]
		doc2 = k[int(k.find("_"))+1:len(k)+1]
		for index,data in enumerate(d):
			vertex = doc1 + '_S' + str(index)
			neighbor_list = []
			if vertex not in page_rank:
				page_rank[vertex] = random.uniform(0,0.5)
			else:
				neighbor_list = neighbors[vertex]
			for ind , data2 in enumerate(data):
				tmp = {}
				if data2 != 0 :
					tmp[doc2 + '_S' + str(ind)] = data2
					neighbor_list.append(tmp)
			neighbors[vertex] = neighbor_list
	#calculate Page Rank
	
	n = len(neighbors.keys())
	# print(n)
	d = 0.85
	epsilon = 0.0000008
	replit = 10
	new_page_rank = {}
	for r in range(replit):
		# print('Page rank repets : %d' % r , end='\r')
		convergence = 0
		for vi , pr_vi in page_rank.items():
			# print("Vi", vi, pr_vi)
			sum_vi = 0
			for n_vi in neighbors[vi]:
				# print("n_vi", n_vi)
				if bool(n_vi):
					vj = list(n_vi.keys())[0]
					dj = list(n_vi.values())[0]
					sim_vj_vz = 0.0
					for n_vj in neighbors[vj]:
						# print("n_vj", n_vj)
						if bool(n_vj):
							vz = list(n_vj.keys())[0]
							dz = list(n_vj.values())[0]
							sim_vj_vz += dz
							# print(vi,vj,dj,vz,dz, sim_vj_vz)
					sum_vi += (dj / sim_vj_vz) * page_rank[vj]
					# print(sum_vi)
			# print("*******************************")
			new_page_rank[vi] = (1 - d) / n + d * sum_vi
		norm = sum(new_page_rank.values())
		for vi , PR_vi in new_page_rank.items():
			new_page_rank[vi] = PR_vi / norm
			convergence += (page_rank[vi] - new_page_rank[vi]) **2 
		if convergence <= epsilon:
			break
		else:
			page_rank = new_page_rank.copy()
	# print()
	return page_rank


basePath = os.getcwd()
path = sys.argv[1]
# nonRePath = os.path.join(basePath, sys.argv[2])
nonRePath = sys.argv[2]
os.chdir(path)
path = os.getcwd()
doc_list = []

topics = topic_signature(path, nonRePath)

os.chdir(path)
for file in os.listdir():
	if os.path.isfile(file):
		doc_list.append(file)
x=0
sumup=0
summ_doc={}
summ_dict={}
harmunic_dict={}
doc_list_size = len(doc_list)
for file1 in doc_list:
	f1 = open(f"{path}/{file1}")
	fr1 = f1.read()
	for file2 in doc_list:
		f2 = open(f"{path}/{file2}")
		fr2 = f2.read()
		l1, l2, l3, l4 ,l5 = similarity(fr1,fr2, topics)
		# print("similarity time : {}".format(time() - ts2))
		list_array=[]
		list_h=[]
		for i in range(0 , len(l1)):
			harmunic_list=[]
			a=0
			b=0
			c=0
			d=0
			f=0
			h=0
			for j in range(0, len(l1[i])):
				if (l1[i][j]==0):
					a=0
				else:
					a=1/l1[i][j]
				if (l2[i][j]==0):
					b=0
				else:
					b=1/l2[i][j]
				if (l3[i][j]==0):
					c=0
				else:
					c=1/l3[i][j]
				if (l4[i][j]==0):
					d=0
				else:
					d=1/l4[i][j]
				if (l5[i][j]==0):
					f=0
				else:
					f=1/l5[i][j]
				if (a+b+c+d+f==0):
					h=0
				else:
					h=5/(a+b+c+d+f)
				harmunic_list.append(h)
			list_h.append(harmunic_list)
		name= file1+"_"+file2
		harmunic_dict[name]=list_h
Harmonic_between_2_algo={}
PR_final_list=pageRank(harmunic_dict)
new_HITS_normalize=hits(harmunic_dict)
for i, j in PR_final_list.items():
	l = new_HITS_normalize[i]
	if j == 0 :
		n = 0.0 + (1/float(l))
	elif l == 0 :
		n=(1/float(j)) + 0.0
	else :
		n=(1/float(j))+(1/float(l))
	Harmonic_between_2_algo[i]=2/n


n_sentns = 15
harmonic_sorted = sorted(Harmonic_between_2_algo.items(), key=lambda x: x[1], reverse=True)
# print(harmonic_sorted)
list_summary=[]
for i in range(0, n_sentns):
	k,d = harmonic_sorted[i]
	doc_name = k[0:int(k.find("_"))]
	sent_numb = k[int(k.find("_"))+2:len(k)]
	f1 = open(f"{path}/{doc_name}").read()
	# print (doc_name , sent_numb , d)
	s=get_sentens(f1,sent_numb)
	# print("sentence:",s)
	if (len(list_summary)==0):
		list_summary.append(s)
		# print("list:",list_summary)
	elif (bow_cosine(s,list_summary) < 0.7):
		list_summary.append(s)
print(list_summary)



