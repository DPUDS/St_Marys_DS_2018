#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 20:00:59 2018

@author: ducnguyen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
from sklearn.metrics import accuracy_score
from sklearn import preprocessing, cross_validation, neighbors
from sklearn.neural_network import MLPClassifier
from datetime import datetime, timedelta


def readData1(numRows = None):
    data = pd.read_table('training_data.txt', nrows = numRows)
    return data

def readData2(numRows = None):
    data = pd.read_table('test_data.txt',nrows = numRows)
    return data


    
def changeCategory(x):
    if x == 'admn': 
        return '1'
    elif x == 'disch' : 
        return '2'
    elif x == 'issues' :
        return '3'
    elif x == 'meals' :
        return '4'
    elif x == 'nurses' : 
        return '5'
    elif x == 'overall' :
        return '6'
    elif x == 'physn':
        return '7'
    elif x == 'room' :
        return '8'
    elif x == 'tests' :
        return '9'
    elif x == 'visit' :
        return '10'
    
    
def main():
  #  df = readData()
    
  #  catTry(df)
    
  #  print('\n')
    
  #  ageSexTry(df)
    
    df2 = readData1()

    
   # print(X)
    #print(lm.intercept_)

    df2.loc[:,'comment']=df2.loc[:,'comment'].map(lambda x: TextBlob(x).sentiment.polarity)
    df2.loc[:,'category']=df2.loc[:,'category'].map(lambda x:changeCategory(x))
    df2.loc[:,'er']=df2.loc[:,'er'].map(lambda x: None if x =='None' else x)
    df2.loc[:,'stay_cat']=df2.loc[:,'stay_cat'].map(lambda x: None if x =='None' else x)
    df2.loc[:,'survey_number']=df2.loc[:,'survey_number'].map(lambda x: None if x =='None' else x)
    df2.loc[:,'sex']=df2.loc[:,'sex'].map(lambda x: None if x =='None' else x)
    df2.loc[:,'age_cat']=df2.loc[:,'age_cat'].map(lambda x: None if x =='None' else x)
    df2.loc[:,'lang']=df2.loc[:,'lang'].map(lambda x: None if x =='None' else x)
    #df2.loc[:,'category']=df2.loc[:,'category'].map(lambda x: None if x =='None' else x)
    #df2.loc[:,''].map(lambda x: None if x =='None' else x)
#    mean = df2.loc[:,'er'].mean()

    temp = df2
    mode1 = temp.loc[:,'er'].dropna()
    mode1 = mode1.map(lambda x: float(x))
    mean1 = mode1.mean()
    temp.loc[:,'er'] = temp.loc[:,'er'].fillna(mean1)
    
    mode2 = temp.loc[:,'stay_cat'].dropna()
    mode2 = mode2.map(lambda x: float(x))
    mean2 = mode2.mean()
    temp.loc[:,'stay_cat'] = temp.loc[:,'stay_cat'].fillna(mean2)
  
    mode3 = temp.loc[:,'survey_number'].dropna()
    mode3 = mode3.map(lambda x: float(x))
    mean3 = mode3.mean()
    temp.loc[:,'survey_number'] = temp.loc[:,'survey_number'].fillna(mean3)
    
    mode4 = temp.loc[:,'sex'].dropna()
    mode4 = mode4.map(lambda x: float(x))
    mean4 = mode4.mean()
    temp.loc[:,'sex'] = temp.loc[:,'sex'].fillna(mean4)
    
    mode5 = temp.loc[:,'age_cat'].dropna()
    mode5 = mode5.map(lambda x: float(x))
    mean5 = mode5.mean()
    temp.loc[:,'age_cat'] = temp.loc[:,'age_cat'].fillna(mean5)
    
    mode6 = temp.loc[:,'lang'].dropna()
    mode6 = mode6.map(lambda x: float(x))
    mean6 = mode6.mean()
    temp.loc[:,'lang'] = temp.loc[:,'lang'].fillna(mean6)
    #print(temp)

    
    X = temp.drop('id', axis =1)
    X = X.drop('score', axis =1)
 #   mean1=erCol.sum()
   # print(df2)
    Z = readZ()
    Z = Z.drop('id',axis=1)
    scores = readData1().loc[:,'score']
    Y=scores
   
    result = linReg(X,Y,Z)
    print(len(result))
 #   kNearest(X,Y)
    
    f = open('DePauw Team 4 Final.txt','w')
    for x in result:
        f.write('{}\n'.format(x))
    f.close()
    
    print(Z)
    print(Z.iloc[26896])
    print('DONE!')

    
    
def readZ():
    df2=readData2()
    df2.loc[:,'comment']=df2.loc[:,'comment'].map(lambda x: TextBlob(x).sentiment.polarity)
    df2.loc[:,'category']=df2.loc[:,'category'].map(lambda x:changeCategory(x))
    df2.loc[:,'er']=df2.loc[:,'er'].map(lambda x: None if x =='None' else x)
    df2.loc[:,'stay_cat']=df2.loc[:,'stay_cat'].map(lambda x: None if x =='None' else x)
    df2.loc[:,'survey_number']=df2.loc[:,'survey_number'].map(lambda x: None if x =='None' else x)
    df2.loc[:,'sex']=df2.loc[:,'sex'].map(lambda x: None if x =='None' else x)
    df2.loc[:,'age_cat']=df2.loc[:,'age_cat'].map(lambda x: None if x =='None' else x)
    df2.loc[:,'lang']=df2.loc[:,'lang'].map(lambda x: None if x =='None' else x)
    #df2.loc[:,'category']=df2.loc[:,'category'].map(lambda x: None if x =='None' else x)
    #df2.loc[:,''].map(lambda x: None if x =='None' else x)
#    mean = df2.loc[:,'er'].mean()

    temp = df2
    mode1 = temp.loc[:,'er'].dropna()
    mode1 = mode1.map(lambda x: float(x))
    mean1 = mode1.mean()
    temp.loc[:,'er'] = temp.loc[:,'er'].fillna(mean1)
    
    mode2 = temp.loc[:,'stay_cat'].dropna()
    mode2 = mode2.map(lambda x: float(x))
    mean2 = mode2.mean()
    temp.loc[:,'stay_cat'] = temp.loc[:,'stay_cat'].fillna(mean2)
  
    mode3 = temp.loc[:,'survey_number'].dropna()
    mode3 = mode3.map(lambda x: float(x))
    mean3 = mode3.mean()
    temp.loc[:,'survey_number'] = temp.loc[:,'survey_number'].fillna(mean3)
    
    mode4 = temp.loc[:,'sex'].dropna()
    mode4 = mode4.map(lambda x: float(x))
    mean4 = mode4.mean()
    temp.loc[:,'sex'] = temp.loc[:,'sex'].fillna(mean4)
    
    mode5 = temp.loc[:,'age_cat'].dropna()
    mode5 = mode5.map(lambda x: float(x))
    mean5 = mode5.mean()
    temp.loc[:,'age_cat'] = temp.loc[:,'age_cat'].fillna(mean5)
    
    mode6 = temp.loc[:,'lang'].dropna()
    mode6 = mode6.map(lambda x: float(x))
    mean6 = mode6.mean()
    temp.loc[:,'lang'] = temp.loc[:,'lang'].fillna(mean6)
    #print(temp)
    return df2
   
    
    
    '''
    lm = LinearRegression()
    print(X)
    lm.fit(X,df2.score)
    print(lm.intercept_)
    print(len(lm.coef_))
      
    #pd.DataFrame(list(zip(X.columns,lm.coef_)), columns = ['features','estimated coefficients'])
   
    prediction = lm.predict(X)
    prediction = prediction.astype(int)
 #  prediction = prediction.map(lambda x: int(x))
 '''



def ageSexTry(df):
    print('Age Sex Try')
   # print(df)
    print('Age Category')
    mean1 = df[df.loc[:, "age_cat"] == "1.0"].loc[:,"score"].mean()
    median1 = df[df.loc[:, "age_cat"] == "1.0"].loc[:,"score"].mode()
    print('The mean of age_cat1.0 is ' +str(mean1))
    print('The mode of age_cat1.0 is ' +str(median1))
    
    mean2 = df[df.loc[:, "age_cat"] == "2.0"].loc[:,"score"].mean()
    median2 = df[df.loc[:, "age_cat"] == "2.0"].loc[:,"score"].mode()
    print('The mean of age_cat2.0 is ' +str(mean2))
    print('The mode of age_cat2.0 is ' +str(median2))
    
    mean3 = df[df.loc[:, "age_cat"] == "3.0"].loc[:,"score"].mean()
    median3 = df[df.loc[:, "age_cat"] == "3.0"].loc[:,"score"].mode()
    print('The mean of age_cat3.0 is ' +str(mean3))
    print('The mode of age_cat3.0 is ' +str(median3))
    
    mean4 = df[df.loc[:, "age_cat"] == "4.0"].loc[:,"score"].mean()
    median4 = df[df.loc[:, "age_cat"] == "4.0"].loc[:,"score"].mode()
    print('The mean of age_cat4.0 is ' +str(mean4))
    print('The mode of age_cat4.0 is ' +str(median4))
    
    mean5 = df[df.loc[:, "age_cat"] == "5.0"].loc[:,"score"].mean()
    median5 = df[df.loc[:, "age_cat"] == "5.0"].loc[:,"score"].mode()
    print('The mean of age_cat5.0 is ' +str(mean5))
    print('The mode of age_cat5.0 is ' +str(median5))
    
    mean6 = df[df.loc[:, "age_cat"] == "6.0"].loc[:,"score"].mean()
    median6 = df[df.loc[:, "age_cat"] == "6.0"].loc[:,"score"].mode()
    print('The mean of age_cat6.0 is ' +str(mean6))
    print('The mode of age_cat6.0 is ' +str(median6))
    
    print('\nSex')
    
    meanF = df[df.loc[:, "sex"] == "2.0"].loc[:,"score"].mean()
    medianF = df[df.loc[:, "sex"] == "2.0"].loc[:,"score"].mode()
    print('The mean of Female is ' +str(meanF))
    print('The mode of Female is ' +str(medianF))
    
    meanM = df[df.loc[:, "sex"] == "1.0"].loc[:,"score"].mean()
    medianM = df[df.loc[:, "sex"] == "1.0"].loc[:,"score"].mode()
    print('The mean of Male is ' +str(meanM))
    print('The mode of Male is ' +str(medianM))
    
def catTry(df):
    print('Category Try: ')
    cat = ['admn','disch','issues','meals','nurses','overall','physn','room','tests','visit']
    for item in cat:
        meanA = df[df.loc[:, "category"] == item].loc[:,"score"].mean()
        medianA = df[df.loc[:, "category"] == item].loc[:,"score"].mode()
        print('The mean for ' + item +' is '+ str(meanA))
        print('The mode for ' + item +' is '+ str(medianA))
        
    
def evaluation(prediction,score):
    er1 = prediction - score
    #print(er1)
    er2 = er1**2
   # print(er2)
    return er2.sum()
   # agecat5plot= agecat5['score'].plot()
   ## plt.show()
    #print(agecat5plot)
    '''  
def kNearest(X,Y):
    clf = neighbors.KNeighborsClassifier()
 
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X,Y,test_size=0.1)
    clf.fit(X_train, Y_train)
    
    prediction = clf.predict(X_test)
        
    print('KNN Error: ',evaluation(prediction,Y_test))
    print('KNN Acc: ',accuracy_score(np.array(Y_test),np.array(prediction)))
    '''
def linReg(X,Y,Z):
    lm = LinearRegression()
    
    lm.fit(X, Y)
    
    
      
    #pd.DataFrame(list(zip(X.columns,lm.coef_)), columns = ['features','estimated coefficients'])
    '''
    plt.scatter(temp.comment,temp.score)
    plt.xlabel('Sentiment of Comments')
    plt.ylabel('Satisfactionary Scores')
    plt.title('Relationship btw Comments and Scores')
    plt.show()
    '''
    prediction = lm.predict(Z)
    prediction = prediction.astype(int)
   # print('Lin Reg Error: ',evaluation(prediction,Y_test))
   # print('Lin Reg Acc: ',accuracy_score(np.array(Y_test),np.array(prediction)))
    return prediction
 #  prediction = prediction.map(lambda x: int(x))
   # accuracy = clf.score(X_test, Y_test)
    
    #clf.fit(X,Y)
   # accuracy1 = clf.score(X,Y)
  #  return clf.predict(X.test)