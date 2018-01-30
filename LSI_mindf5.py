# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 22:47:59 2018

@author: Firnaz
"""

#Importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import classification_report

#Importing the datasets
c = ['comp.graphics','comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware','rec.autos', 'rec.motorcycles',
              'rec.sport.baseball', 'rec.sport.hockey']
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

#Plotting the histogram with different categories
bins = np.arange(9) - 0.5
plt.hist(training_set.target, bins = bins, alpha=0.7)
plt.xlabel('Target output class label')
plt.ylabel('Count of documents')
plt.title('Histogram of documents in each category')
plt.show()

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

###########################################################################                    
#Using min_df = 5       
###########################################################################

# Creating the Bag of Words model (min_df = 5)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(min_df=5)
count_matrix_train = cv.fit_transform(train_corpus).toarray()
count_matrix_test = cv.transform(test_corpus).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
tfidf_matrix_train = tfidf.fit_transform(count_matrix_train).toarray()
tfidf_matrix_test = tfidf.transform(count_matrix_test).toarray()

#Using LSI for dimensionality reduction
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components = 50, random_state = 0)
x_train = svd.fit_transform(tfidf_matrix_train)
x_test = svd.transform(tfidf_matrix_test)

#Creating the training and testing data output
y_train = np.zeros(len(train_target))
for i in range(len(train_target)):
    if train_target['class'][i] > 3:
        y_train[i] = 1
    else:
        y_train[i] = 0
y_test = np.zeros(len(test_target))
for i in range(len(test_target)):
    if test_target['class'][i] > 3:
        y_test[i] = 1
    else:
        y_test[i] = 0

# Fitting Hard Margin SVM to the Training set
from sklearn.svm import SVC
classifier_hm = SVC(kernel = 'linear', random_state = 0, C=1000, probability = True)
classifier_hm.fit(x_train, y_train)

# Predicting the Test set results
y_pred_hm = classifier_hm.predict(x_test)
y_pred_hm_prob = classifier_hm.predict_proba(x_test)[:,1]

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_hm = confusion_matrix(y_test, y_pred_hm)    

#Calculating accuracy, precision and recall 
accuracy_hm = (cm_hm[0,0] + cm_hm[1,1])/cm_hm.sum()
precision_hm =  cm_hm[1,1] / (cm_hm[1,1] + cm_hm[0,1])
recall_hm = cm_hm[1,1] / (cm_hm[1,1] + cm_hm[1,0])
print(classification_report(y_test, y_pred_hm))

#Plotting the ROC 
from sklearn.metrics import roc_curve
fpr, tpr, threshold = roc_curve(y_test, y_pred_hm_prob)
plt.plot(fpr,tpr)
plt.title('ROC curve for Hard Margin Classifier with C=1000')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.ylim(0,1.1)
plt.show()

# Fitting Soft Margin SVM to the Training set
from sklearn.svm import SVC
classifier_sm = SVC(kernel = 'linear', random_state = 0, C=0.001, probability = True)
classifier_sm.fit(x_train, y_train)

# Predicting the Test set results
y_pred_sm = classifier_sm.predict(x_test)
y_pred_sm_prob = classifier_sm.predict_proba(x_test)[:,1]

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_sm = confusion_matrix(y_test, y_pred_sm)    

#Calculating accuracy, precision and recall 
accuracy_sm = (cm_sm[0,0] + cm_sm[1,1])/cm_sm.sum()
precision_sm =  cm_sm[1,1] / (cm_sm[1,1] + cm_sm[0,1])
recall_sm = cm_sm[1,1] / (cm_sm[1,1] + cm_sm[1,0])
print(classification_report(y_test, y_pred_sm))

#Plotting the ROC 
from sklearn.metrics import roc_curve
fpr, tpr, threshold = roc_curve(y_test, y_pred_sm_prob)
plt.plot(fpr,tpr)
plt.title('ROC curve for Soft Margin Classifier with C=0.001')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.ylim(0,1.1)
plt.show()

# Applying k-Fold Cross Validation
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.001,0.01,0.1,1,10,100,1000]}
clf = GridSearchCV(estimator = classifier_hm,param_grid = param_grid, cv=5)
clf.fit(x_train,y_train)
clf.cv_results_
clf.best_estimator_

# Fitting the best SVM to the Training set
from sklearn.svm import SVC
classifier_best = SVC(kernel = 'linear', random_state = 0, C=100, probability = True)
classifier_best.fit(x_train, y_train)

# Predicting the Test set results
y_pred_best = classifier_best.predict(x_test)
y_pred_best_prob = classifier_best.predict_proba(x_test)[:,1]

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_best = confusion_matrix(y_test, y_pred_best)    

#Calculating accuracy, precision and recall 
accuracy_best = (cm_best[0,0] + cm_best[1,1])/cm_best.sum()
precision_best =  cm_best[1,1] / (cm_best[1,1] + cm_best[0,1])
recall_best = cm_best[1,1] / (cm_best[1,1] + cm_best[1,0])
print(classification_report(y_test, y_pred_best))

#Plotting the ROC 
from sklearn.metrics import roc_curve
fpr, tpr, threshold = roc_curve(y_test, y_pred_best_prob)
plt.plot(fpr,tpr)
plt.title('ROC curve for SVM Classifier with C=100')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.ylim(0,1.1)
plt.show()

#Naive Bayes Classifier
#Fitting the Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
classifier_b = GaussianNB()
classifier_b.fit(x_train, y_train)

# Predicting the Test set results
y_pred_b = classifier_b.predict(x_test)
y_pred_b_prob = classifier_b.predict_proba(x_test)[:,1]

# Making the Confusion Matrix
cm_b = confusion_matrix(y_test, y_pred_b)

#Calculating accuracy, precision and recall 
accuracy_b = (cm_b[0,0] + cm_b[1,1])/cm_b.sum()
precision_b =  cm_b[1,1] / (cm_b[1,1] + cm_b[0,1])
recall_b = cm_b[1,1] / (cm_b[1,1] + cm_b[1,0])

#Plotting the ROC 
fpr, tpr, threshold = roc_curve(y_test, y_pred_b_prob)
plt.plot(fpr,tpr)
plt.title('ROC curve for Naive Bayes Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.ylim(0,1.1)
plt.show()

#Logistic Regression Classifier
#Fitting the logistic regression classifier 
from sklearn.linear_model import LogisticRegression
classifier_l = LogisticRegression(random_state = 0)
classifier_l.fit(x_train, y_train)

# Predicting the Test set results
y_pred_l = classifier_l.predict(x_test)
y_pred_l_prob = classifier_l.predict_proba(x_test)[:,1]
cm_l = confusion_matrix(y_test, y_pred_l)

#Calculating accuracy, precision and recall 
accuracy_l = (cm_l[0,0] + cm_l[1,1])/cm_l.sum()
precision_l =  cm_l[1,1] / (cm_l[1,1] + cm_l[0,1])
recall_l = cm_l[1,1] / (cm_l[1,1] + cm_l[1,0])

#Plotting the ROC 
fpr, tpr, threshold = roc_curve(y_test, y_pred_l_prob)
plt.plot(fpr,tpr)
plt.title('ROC curve for Logistic Regression Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

#Using regularization in Logistic regression
parameters = [{'C': [0.001,0.01,0.1,1, 10, 100, 1000], 'penalty': ['l2']},
               {'C': [0.001,0.01,0.1,1, 10, 100, 1000], 'penalty': ['l1']}]
grid_search = GridSearchCV(estimator = classifier_l, param_grid = parameters, cv = 5)
grid_search = grid_search.fit(x_train, y_train)
grid_search.best_score_
grid_search.best_params_
grid_search.cv_results_

#Logistic Regression Classifier with penalty l2 and C=100
#Fitting the logistic regression classifier 
from sklearn.linear_model import LogisticRegression
classifier_bestl2 = LogisticRegression(penalty = 'l2', C=100, random_state = 0)
classifier_bestl2.fit(x_train, y_train)

# Predicting the Test set results
y_pred_bestl2 = classifier_bestl2.predict(x_test)
y_pred_bestl2_prob = classifier_bestl2.predict_proba(x_test)[:,1]

#Plotting the ROC 
fpr, tpr, threshold = roc_curve(y_test, y_pred_bestl2_prob)
plt.plot(fpr,tpr)
plt.title('ROC curve for Logistic Regression Classifier with l2 penalty and C=100')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.ylim(0,1.1)
plt.show()

#Logistic Regression Classifier with penalty l1 and C=10
#Fitting the logistic regression classifier 
from sklearn.linear_model import LogisticRegression
classifier_bestl1 = LogisticRegression(penalty = 'l1', C=10, random_state = 0)
classifier_bestl1.fit(x_train, y_train)

# Predicting the Test set results
y_pred_bestl1 = classifier_bestl1.predict(x_test)
y_pred_bestl1_prob = classifier_bestl1.predict_proba(x_test)[:,1]

#Plotting the ROC 
fpr, tpr, threshold = roc_curve(y_test, y_pred_bestl1_prob)
plt.plot(fpr,tpr)
plt.title('ROC curve for Logistic Regression Classifier with l1 penalty and C=10')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.ylim(0,1.1)
plt.show()