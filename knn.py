# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

#import libraries

import numpy as np    # import mathematical computations, funciotns etc
import matplotlib.pyplot  as plt  # imports necessery functions for charts
import pandas as pd
# import dataset

dataset=pd.read_csv('Social_Network_Ads.csv')  # write sep if in csv file column seperated by semi colon
X=dataset.iloc[:,[2,3]].values             #X (independant var) value capital letter,  ':'- means all values
y=dataset.iloc[:, 4].values              # y (dependant var) -small letter

#split data set into training set and test set

from sklearn.cross_validation import train_test_split
X_train,X_test, y_train, y_test =train_test_split(X,y,test_size=0.25, random_state= 0) # test size 0.2=20%

#Feature scaling  -standarization
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#,sep=";"   - if columna are sperated by semi-colon.

# Fitting Classifier (K nearest neighbour) to the training set  

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,p=2, metric='minkowski')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred=classifier.predict(X_test)

#Making the Confusuin Matrix

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

