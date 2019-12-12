# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 17:33:51 2019

@author: Agatka
"""
#import libraries

import numpy as np    # import mathematical computations, funciotns etc
import matplotlib.pyplot  as plt  # imports necessery functions for charts
import pandas as pd

dataset=pd.read_csv('Data.csv',sep=";")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:, 3].values

from sklearn.preprocessing import Imputer # import library 
imputer=Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer=imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:, 1:3])

imputer.transform(X[:, 1:3])

from sklearn.preprocessing import LabelEncoder  # import library to change categorical variable into numerical

labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])