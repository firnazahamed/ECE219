# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 14:24:56 2018

@author: Firnaz
"""

#Importing the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import NMF
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

#Importing the datasets
c = ['comp.graphics','comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware','rec.autos', 'rec.motorcycles',
              'rec.sport.baseball', 'rec.sport.hockey']
dataset = fetch_20newsgroups(subset = 'all', categories = c, shuffle = True, random_state = 0)

#Convert the dataset into a pandas dataframe
data = dataset.data
target = dataset.target
y = [1 if target[i]>3 else 0 for i in range(len(target))]

# Cleaning the texts
from sklearn.feature_extraction import text
stop_words = text.ENGLISH_STOP_WORDS
import re
corpus = []
for i in range(len(data)):
    text = re.sub('[^a-zA-Z]', ' ', data[i])
    text = text.lower()
    text = text.split()
    text = [word for word in text if not word in stop_words]
    text = ' '.join(text)
    corpus.append(text)   

##############################################################################
#Question 1
##############################################################################
                     
# Creating the TFIDF model (min_df=3)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=3)
tfidf_matrix = tfidf.fit_transform(corpus).toarray()

##############################################################################
#Question 2
##############################################################################

#Fitting K-means and predicting the output class
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 0, n_init=1)
y_kmeans = kmeans.fit_predict(tfidf_matrix)

#Performance Metrics
print('Contingency table:\n', confusion_matrix(y, y_kmeans) ) 
print("Homogeneity: %0.4f" % metrics.homogeneity_score(y, y_kmeans))
print("Completeness: %0.4f" % metrics.completeness_score(y, y_kmeans))
print("V-measure: %0.4f" % metrics.v_measure_score(y, y_kmeans))
print("Adjusted Rand-Index: %.4f"
      % metrics.adjusted_rand_score(y, y_kmeans))
print("Adjusted Mutual Info score: %.4f"
      % metrics.adjusted_mutual_info_score(y, y_kmeans))

##############################################################################
#Question 3b
##############################################################################

#To find the best r in LSI dimensionality reduction
r = [1,2,3,5,10,20,50,100,300]
cm = []
homogenity = []
completeness = []
v_measure = [] 
rand = []
mutual_info = []
for num_comp in r:
    nmf = NMF(n_components = num_comp, random_state = 0)
    nmf_matrix = nmf.fit_transform(tfidf_matrix)
    kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 0, n_init=1)
    y_kmeans = kmeans.fit_predict(nmf_matrix)
    A = confusion_matrix(y, y_kmeans)  
    cm.append(A)
    homogenity.append(metrics.homogeneity_score(y, y_kmeans))
    completeness.append(metrics.completeness_score(y, y_kmeans))
    v_measure.append(metrics.v_measure_score(y, y_kmeans))
    rand.append(metrics.adjusted_rand_score(y, y_kmeans))
    mutual_info.append(metrics.adjusted_mutual_info_score(y, y_kmeans))
    
#Plotting the various accuracy measure 
plt.scatter(x= r,y= homogenity)
plt.ylabel('Homogenity')
plt.xlabel('r Principal components')
plt.title('Homogenity for different r values')
plt.show()

plt.scatter(x= r,y= completeness)
plt.ylabel('Completeness')
plt.xlabel('r Principal components')
plt.title('Completeness for different r values')
plt.show()

plt.scatter(x= r,y= v_measure)
plt.ylabel('V-Measure')
plt.xlabel('r Principal components')
plt.title('V-Measure for different r values')
plt.show()

plt.scatter(x= r,y= rand)
plt.ylabel('Rand')
plt.xlabel('r Principal components')
plt.title('Rand score for different r values')
plt.show()

plt.scatter(x= r,y= mutual_info)
plt.ylabel('Mutual info')
plt.xlabel('r Principal components')
plt.title('Mutual info score for different r values')
plt.show()

##############################################################################
#Question 4a
##############################################################################
#Visualising for the best r
measures = [homogenity,completeness,v_measure,rand,mutual_info]
l = [np.argmax(i) for i in measures]
best_r = r[max(set(l), key=l.count)]
nmf = NMF(n_components = best_r, random_state = 0)
best_reduced_matrix = nmf.fit_transform(tfidf_matrix)
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 0, n_init=5)
y_kmeans = kmeans.fit_predict(best_reduced_matrix)

nmf = NMF(n_components = 2, random_state = 0)
twoD_matrix = nmf.fit_transform(tfidf_matrix)
plt.scatter(twoD_matrix[:,0],twoD_matrix[:,1],c = y_kmeans)
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('Clustering results')
plt.show()

##############################################################################
#Question 4b
##############################################################################

#Normalizing the features
scaler = StandardScaler()
best_scaled_matrix = scaler.fit_transform(best_reduced_matrix)

#Applying clustering on the scaled matrix
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42, n_init=5)
y_kmeans = kmeans.fit_predict(best_scaled_matrix)  

#Visualising the results 
plt.scatter(twoD_matrix[:,0],twoD_matrix[:,1],c = y_kmeans)
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('Clustering results after normalizing')
plt.show()

#Getting the accuracy measures
#Performance Metrics
print('The contingency table is:\n', confusion_matrix(y, y_kmeans) ) 
print("Homogeneity: %0.4f" % metrics.homogeneity_score(y, y_kmeans))
print("Completeness: %0.4f" % metrics.completeness_score(y, y_kmeans))
print("V-measure: %0.4f" % metrics.v_measure_score(y, y_kmeans))
print("Adjusted Rand-Index: %.4f"
      % metrics.adjusted_rand_score(y, y_kmeans))
print("Adjusted Mutual Info score: %.4f"
      % metrics.adjusted_mutual_info_score(y, y_kmeans))

##############################################################################
#Question 4b(ii)
##############################################################################

best_log_matrix = np.log1p(best_reduced_matrix)

#Applying clustering on the scaled matrix
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 0, n_init=5)
y_kmeans = kmeans.fit_predict(best_log_matrix)  

#Visualising the results 
plt.scatter(twoD_matrix[:,0],twoD_matrix[:,1],c = y_kmeans)
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('Clustering results after log transformation')
plt.show()

#Performance Metrics
print('The contingency table is:\n', confusion_matrix(y, y_kmeans) ) 
print("Homogeneity: %0.4f" % metrics.homogeneity_score(y, y_kmeans))
print("Completeness: %0.4f" % metrics.completeness_score(y, y_kmeans))
print("V-measure: %0.4f" % metrics.v_measure_score(y, y_kmeans))
print("Adjusted Rand-Index: %.4f"
      % metrics.adjusted_rand_score(y, y_kmeans))
print("Adjusted Mutual Info score: %.4f"
      % metrics.adjusted_mutual_info_score(y, y_kmeans))


##############################################################################
#Question 4b(iii) Normalize and then log transform
##############################################################################
#Normalizing the features
scaler = StandardScaler()
best_scaled_matrix = scaler.fit_transform(best_reduced_matrix)
best_scaled_matrix -= best_scaled_matrix.min()
best_log_matrix = np.log1p(best_scaled_matrix)

#Applying clustering on the scaled matrix
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 17, n_init=5)
y_kmeans = kmeans.fit_predict(best_log_matrix)  

#Visualising the results 
plt.scatter(twoD_matrix[:,0],twoD_matrix[:,1],c = y_kmeans)
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('Clustering results after normalizing and then taking log transformation')
plt.show()

#Performance Metrics
print('The contingency table is:\n', confusion_matrix(y, y_kmeans) ) 
print("Homogeneity: %0.4f" % metrics.homogeneity_score(y, y_kmeans))
print("Completeness: %0.4f" % metrics.completeness_score(y, y_kmeans))
print("V-measure: %0.4f" % metrics.v_measure_score(y, y_kmeans))
print("Adjusted Rand-Index: %.4f"
      % metrics.adjusted_rand_score(y, y_kmeans))
print("Adjusted Mutual Info score: %.4f"
      % metrics.adjusted_mutual_info_score(y, y_kmeans))

##############################################################################
#Question 4b(iii) Log transform and then normalize
##############################################################################
#Transform and Normalizing the features
best_log_matrix = np.log1p(best_reduced_matrix)
scaler = StandardScaler()
best_scaled_matrix = scaler.fit_transform(best_log_matrix)

#Applying clustering on the scaled matrix
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 0, n_init=5)
y_kmeans = kmeans.fit_predict(best_scaled_matrix)  

#Visualising the results 
plt.scatter(twoD_matrix[:,0],twoD_matrix[:,1],c = y_kmeans)
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('Clustering results with log transformation and then normalizing')
plt.show()

#Performance Metrics
print('The contingency table is:\n', confusion_matrix(y, y_kmeans) ) 
print("Homogeneity: %0.4f" % metrics.homogeneity_score(y, y_kmeans))
print("Completeness: %0.4f" % metrics.completeness_score(y, y_kmeans))
print("V-measure: %0.4f" % metrics.v_measure_score(y, y_kmeans))
print("Adjusted Rand-Index: %.4f"
      % metrics.adjusted_rand_score(y, y_kmeans))
print("Adjusted Mutual Info score: %.4f"
      % metrics.adjusted_mutual_info_score(y, y_kmeans))
