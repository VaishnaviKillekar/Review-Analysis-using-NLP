# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 16:52:57 2018

@author: Vaishnavi
"""

# NATURAL LANGUAGE PROCESSING
# Restaurant Review processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset     # 3 stands for ignoring double quotes
dataset=pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

# Cleaning the text
import re
import  nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    # Keeping only alphabets
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i] )
    # All aplhabets must be lower case only
    review = review.lower()
    # Split review in different words
    review = review.split()
    # Remove non-significant words     set is used when review is large like an article
    # Stemming - only keep root word
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #  Convert the list review back to string 
    review = ' '.join(review)
    # Append the cleaned review to list corpus
    corpus.append(review)
    
# Creating Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Model Performace
accuracy = (cm[1][1]+cm[0][0])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
precision = cm[1][1]/(cm[0][0]+cm[0][1])
recall = cm[1][1]/(cm[1][1]+cm[1][0])

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)