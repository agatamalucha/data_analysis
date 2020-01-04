
#1. import libraries

import numpy as np    # import mathematical computations, funciotns etc
import matplotlib.pyplot  as plt  # imports necessery functions for charts
import pandas as pd
# import dataset
dataset=pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting= 3)  #quoting- parameter 3 is to ignore dbl quotes



# 2. cleaning the text

import re 
import nltk 
nltk.download('stopwords') 
from nltk.corpus import stopwords                        #library that allows to remove irrelevant words
from nltk.stem.porter import PorterStemmer     #import class to make root of word 
corpus=[]
for  i in range(0,1000):

    review=re.sub('[^a-zA-Z]',' ', dataset['Review'][i])     # removing numbers, exclamation marks etc.1st parameter ^ not to remove any letters ,#2nd parameter -removed charackter will be replaced by space                                                                                       
                                                        #3nrd parameter- what part of dataset to check -first review
                                                         
    review =review.lower()             # replacing Capital letter with lower cases letters                               
    review= review.split()  
    ps=PorterStemmer()
    review= [ps.stem(word) for word in review if not  word in set(stopwords.words('english'))]   #loop to remove all irrelevant words
    review=' '.join(review)       # review replaced back to string joining to sentence 
    corpus.append(review)


# creating bag of words model
    
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values


#Model Naive Bayes
  
#split data set into training set and test set

from sklearn.cross_validation import train_test_split
X_train,X_test, y_train, y_test =train_test_split(X,y,test_size=0.20, random_state= 0) # test size 0.2=20%

# Fitting Naive Bayes to the training set

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred=classifier.predict(X_test)

#Making the Confusuin Matrix

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
  
  

#review = []
 #for word in review:
  #   if not word in set:
    #     word = ps.stem(word)
    #     review.append(word)
         
 
 
 
 
 # to view item in dataset dataset['name of column',[numer of review ]]