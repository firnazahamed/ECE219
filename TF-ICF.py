# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 23:28:59 2018

@author: Firnaz
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 23:06:47 2018

@author: Firnaz
"""

#Importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups

#Importing the datasets
data_set = fetch_20newsgroups(subset = 'all', shuffle = True, random_state = 0)
test_set = fetch_20newsgroups(subset = 'test', shuffle = True, random_state = 0) 

#Convert the training dataset into a pandas dataframe
data = pd.DataFrame(data_set.data)
data_target = pd.DataFrame(data_set.target)
dataset = pd.concat([data,data_target],axis = 1)
dataset.columns = ['text','class']

#Cleaning the texts
from sklearn.feature_extraction import text
stop_words = text.ENGLISH_STOP_WORDS
import re
from nltk.stem.porter import PorterStemmer
corpus = [ [] for _ in range(20)]
for i in range(len(dataset)):
    text = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in stop_words]
    text = ' '.join(text)
    for j in range(20):
        if dataset['class'][i] == j:
            corpus[j].append(text)
corp = []            
for i in range(20):
    docs = corpus[i]
    docs = ' '.join(docs)
    corp.append(docs)
            

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(min_df=2)
count_matrix = cv.fit_transform(corp).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
tfidf_matrix = tfidf.fit_transform(count_matrix).toarray()            
            
feature_names = cv.get_feature_names()
ibm = tfidf_matrix[3,:]
mac = tfidf_matrix[4,:]
misc = tfidf_matrix[6,:]
christian = tfidf_matrix[15,:] 
ibm_top10 = sorted(zip(ibm,feature_names),reverse=True)[:10]
mac_top10 = sorted(zip(mac,feature_names),reverse=True)[:10]
misc_top10 = sorted(zip(misc,feature_names),reverse=True)[:10]
christian_top10 = sorted(zip(christian,feature_names),reverse=True)[:10]
