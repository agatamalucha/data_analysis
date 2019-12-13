# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 17:33:51 2019

@author: Agatka
"""
#import libraries

import numpy as np    # import mathematical computations, funciotns etc
import matplotlib.pyplot  as plt  # imports necessery functions for charts
import pandas as pd

dataset=pd.read_csv('Data.csv',sep=";")  # write sep if in csv file column seperated by semi colon
X=dataset.iloc[:,:-1].values             #X (independant var) value capital letter,  ':'- means all values
y=dataset.iloc[:, 3].values              # y (dependant var) -small letter

from sklearn.preprocessing import Imputer # import library        # to fill out missing variables
imputer=Imputer(missing_values='NaN', strategy='mean', axis=0)  # Nan , 'mean' strategy= average count of cariable
imputer=imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:, 1:3])     y small (depenable var)

imputer.transform(X[:, 1:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # import library to change categorical variable into numerical

labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
onehotencoder=OneHotEncoder(categorical_features= [0])
X=onehotencoder.fit_transform(X).toarray()

#split data set into training set and test set
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)
from sklearn.cross_validation import train_test_split
X_train,X_test, y_train, y_test =train_test_split(X,y,test_size=0.2, random_state= 0) # test size 0.2=20%
