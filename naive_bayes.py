#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 18:39:40 2020

@author: mehdi
"""

# Naive Bayes

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('ar_reviews_100k.tsv', sep='\t')

# Data Cleaning
stopwords = pd.read_csv('ar_stopwords.txt', header = None)
dataset['text'] = dataset['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Applying TF-TDF
from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer()
v.max_features = 5000
X_train = v.fit_transform(X_train).toarray()
X_test = v.transform(X_test).toarray()

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import  MultinomialNB # GaussianNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred, labels=['Positive', 'Mixed', 'Negative'])

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

# Testing with other values  
comments=['هذا رائع',
          'الكتاب ممل',
          'الخدمة سيئة',
          'في حين ان الفندق رائع لكن خدمة الزبائن سيئة و المكان غير نظيف و لكن هذا يتماشى و السعر',
          'افضل الذهاب عند منافسكم',
          'الخدمة بطيئة جدا',
          'احببت نهاية الفلم',
          'السعر مرتفع جدا',
          'لقد ندمت على استخدام منتجكم',
          'شكرا جزيلا على الخدمة الرائعة']
data = [[comments[0],classifier.predict(v.transform(([comments[0]])))],
        [comments[1],classifier.predict(v.transform(([comments[1]])))],
        [comments[2],classifier.predict(v.transform(([comments[2]])))],
        [comments[3],classifier.predict(v.transform(([comments[3]])))],
        [comments[4],classifier.predict(v.transform(([comments[4]])))],
        [comments[5],classifier.predict(v.transform(([comments[5]])))],
        [comments[6],classifier.predict(v.transform(([comments[6]])))],
        [comments[7],classifier.predict(v.transform(([comments[7]])))],
        [comments[8],classifier.predict(v.transform(([comments[8]])))],
        [comments[9],classifier.predict(v.transform(([comments[9]])))]
        ] 
df = pd.DataFrame(data, columns = ['Comment', 'Class']) 
