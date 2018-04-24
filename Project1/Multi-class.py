# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:40:13 2018

@author: Firnaz
"""

#Importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier

#Importing the datasets
c = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale',
     'soc.religion.christian']
training_set = fetch_20newsgroups(subset = 'train', categories = c, shuffle = True, random_state = 0)
test_set = fetch_20newsgroups(subset = 'test', categories = c, shuffle = True, random_state = 0) 

#Convert the training dataset into a pandas dataframe
train_data = pd.DataFrame(training_set.data)
train_target = pd.DataFrame(training_set.target)
train_target.columns = ['class']
train = pd.concat([train_data,train_target],axis = 1)
train.columns = ['text','class']

#Convert the test dataset into a pandas dataframe
test_data = pd.DataFrame(test_set.data)
test_target = pd.DataFrame(test_set.target)
test_target.columns = ['class']
test = pd.concat([test_data,test_target],axis = 1)
test.columns = ['text','class']

#Merging the training and test set for preprocessing
dataset = pd.concat([train, test], axis = 0, ignore_index = True)

# Cleaning the texts
from sklearn.feature_extraction import text
stop_words = text.ENGLISH_STOP_WORDS

import re
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(len(dataset)):
    text = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in stop_words]
    text = ' '.join(text)
    corpus.append(text)
    
train_corpus = corpus[:len(train_data)]
test_corpus = corpus[len(train_data):]

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(min_df=2)
count_matrix_train = cv.fit_transform(train_corpus).toarray()
count_matrix_test = cv.transform(test_corpus).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
tfidf_matrix_train = tfidf.fit_transform(count_matrix_train).toarray()
tfidf_matrix_test = tfidf.transform(count_matrix_test).toarray()

##############################################################################
#LSI
##############################################################################

#Using LSI for dimensionality reduction
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components = 50, random_state = 0)
x_train = svd.fit_transform(tfidf_matrix_train)
x_test = svd.transform(tfidf_matrix_test)

#Creating the training and testing data output
y_train = np.array(train_target)
y_test = np.array(test_target)

#Naive Bayes Classifier
#Fitting the Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
classifier_b = GaussianNB()
classifier_b.fit(x_train, y_train)

# Predicting the Test set results
y_pred_b = classifier_b.predict(x_test)

# Making the Confusion Matrix
cm_b = confusion_matrix(y_test, y_pred_b)

#Calculating accuracy, precision and recall 
accuracy_b = (cm_b[0,0] + cm_b[1,1] + cm_b[2,2] + cm_b[3,3])/cm_b.sum()
print(classification_report(y_test,y_pred_b, digits=4))

#SVM Classifier
# Creating the SVM Classifier
from sklearn.svm import SVC
classifier_svm = SVC(kernel = 'linear', random_state = 0)

#Creating the one vs one classifier
classifier_svm_ovo = OneVsOneClassifier(classifier_svm)
classifier_svm_ovo.fit(x_train,y_train)
# Predicting the Test set results
y_pred_svm_ovo = classifier_svm_ovo.predict(x_test)
# Making the Confusion Matrix, accuracy, precision, recall
cm_svm_ovo = confusion_matrix(y_test, y_pred_svm_ovo)
accuracy_svm_ovo = (cm_svm_ovo[0,0] + cm_svm_ovo[1,1] + cm_svm_ovo[2,2] + cm_svm_ovo[3,3])/cm_svm_ovo.sum()
print(classification_report(y_test, y_pred_svm_ovo,digits=4))
    
#Creating the one vs rest classifier
classifier_svm_ovr = OneVsRestClassifier(classifier_svm)
classifier_svm_ovr.fit(x_train,y_train)
# Predicting the Test set results
y_pred_svm_ovr = classifier_svm_ovr.predict(x_test)
# Making the Confusion Matrix, accuracy, precision, recall
cm_svm_ovr = confusion_matrix(y_test, y_pred_svm_ovr)
accuracy_svm_ovr = (cm_svm_ovr[0,0] + cm_svm_ovr[1,1] + cm_svm_ovr[2,2] + cm_svm_ovr[3,3])/cm_svm_ovr.sum()
print(classification_report(y_test, y_pred_svm_ovr,digits=4))

##############################################################################
#NMF
##############################################################################

#Using NMF for dimensionality reduction
from sklearn.decomposition import NMF
svd = NMF(n_components = 50, random_state = 0)
x_train = svd.fit_transform(tfidf_matrix_train)
x_test = svd.transform(tfidf_matrix_test)

#Creating the training and testing data output
y_train = np.array(train_target)
y_test = np.array(test_target)

#Naive Bayes Classifier
#Fitting the Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
classifier_b = MultinomialNB()
classifier_b.fit(x_train, y_train)

# Predicting the Test set results
y_pred_b = classifier_b.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_b = confusion_matrix(y_test, y_pred_b)

#Calculating accuracy, precision and recall 
accuracy_b = (cm_b[0,0] + cm_b[1,1] + cm_b[2,2] + cm_b[3,3])/cm_b.sum()
print(classification_report(y_test, y_pred_b, digits=4))

#SVM Classifier
# Creating the SVM Classifier
from sklearn.svm import SVC
classifier_svm = SVC(kernel = 'linear', random_state = 0)

#Creating the one vs one classifier
classifier_svm_ovo = OneVsOneClassifier(classifier_svm)
classifier_svm_ovo.fit(x_train,y_train)
# Predicting the Test set results
y_pred_svm_ovo = classifier_svm_ovo.predict(x_test)
# Making the Confusion Matrix, accuracy, precision, recall
cm_svm_ovo = confusion_matrix(y_test, y_pred_svm_ovo)
accuracy_svm_ovo = (cm_svm_ovo[0,0] + cm_svm_ovo[1,1] + cm_svm_ovo[2,2] + cm_svm_ovo[3,3])/cm_svm_ovo.sum()
print(classification_report(y_test, y_pred_svm_ovo, digits=4))
    
#Creating the one vs rest classifier
classifier_svm_ovr = OneVsRestClassifier(classifier_svm)
classifier_svm_ovr.fit(x_train,y_train)
# Predicting the Test set results
y_pred_svm_ovr = classifier_svm_ovr.predict(x_test)
# Making the Confusion Matrix, accuracy, precision, recall
cm_svm_ovr = confusion_matrix(y_test, y_pred_svm_ovr)
accuracy_svm_ovr = (cm_svm_ovr[0,0] + cm_svm_ovr[1,1] + cm_svm_ovr[2,2] + cm_svm_ovr[3,3])/cm_svm_ovr.sum()
print(classification_report(y_test, y_pred_svm_ovr, digits=4))
