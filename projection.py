#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 11:33:10 2017

@author: Abhay
"""
import numpy as np
import csv as csv
import pandas as pd
import os
import scipy as sp
import math
from sklearn.cross_validation import train_test_split
from scipy import linalg
from datetime import datetime

def date_diff(certified, operational):
    certified = datetime.strptime(certified, '%d/%m/%Y')
    operational = datetime.strptime(operational, '%d/%m/%Y')
    if (operational - certified) >= 0:
        return 1
    else:
        return 0

def array2csv(array, filename):
    
    output_Str = ''
    
    for eachrow in array:
        row_arr = np.asarray(eachrow)
        row_Str = ','.join(map(str,row_arr))
        output_Str = output_Str + row_Str + '\n'


    output_filename = open(filename, 'w')

    output_filename.write(output_Str)


def class2csv(array, filename):
    
    output_Str = '\n'.join(map(str,array))

    output_filename = open(filename, 'w')

    output_filename.write(output_Str)  

    
for i in range(newfeatures.shape[0]):
    if newfeatures[i,18] == str("Yes"):
        newfeatures[i,18] = 0
    else:
        newfeatures[i,18] = 1
        
    if newfeatures[i,22] == str("AM"):
        newfeatures[i,22] = 0
    elif newfeatures[i,22] == str("ASPA"):
        newfeatures[i,22] = 1
    elif newfeatures[i,22] == str("EURO"):
        newfeatures[i,22] = 2
    elif newfeatures[i,22] == str("MEA"):
        newfeatures[i,22] = 3
    else:
        newfeatures[i,22] = 4
    
    if newfeatures[i,23] == str("AM"):
        newfeatures[i,23] = 0
    elif newfeatures[i,23] == str("ASPA"):
        newfeatures[i,23] = 1
    elif newfeatures[i,23] == str("EURO"):
        newfeatures[i,23] = 2
    elif newfeatures[i,23] == str("MEA"):
        newfeatures[i,23] = 3
    else:
        newfeatures[i,23] = 4
        
    if newfeatures[i,24] or newfeatures[i,25]:
        newfeatures[i,24] = 0
    else:
        newfeatures[i,24] = date_diff(newfeatures[i,25], newfeatures[i,24])
        

normalized_features = pd.read_csv('/Users/Abhay/Desktop/DHL/normalized_features.csv')
normalized_feature_arr = np.asarray(normalized_features)

normalized_feature_arr = newfeatures[:,0:25]

#features = np.delete(normalized_feature_arr,[0,3,4,5,15],axis=1)
features = np.delete(normalized_feature_arr,[0,3,4,5,15,17,19],axis=1)
X_train, X_test, y_train, y_test = train_test_split(features, normalized_feature_arr[:,15], test_size=.1)

x_c1_feature = X_train[y_train==0,:]
x_c2_feature = X_train[y_train==1,:]

mean_x_c1_feature = x_c1_feature - x_c1_feature.mean()
tran_mean = mean_x_c1_feature.T
prod = np.inner(tran_mean , tran_mean)
auto_x_c1_feature = (prod)/np.std(prod)

mean_x_c2_feature = x_c2_feature - x_c2_feature.mean()
tran_mean = mean_x_c2_feature.T
prod = np.inner(tran_mean , tran_mean)
auto_x_c2_feature = (prod)/np.std(prod)

a1= np.asarray(auto_x_c1_feature).astype(None)
b1=np.asarray(auto_x_c2_feature).astype(None)

D, V = linalg.eig(a1, b=b1)

CSP = []
D_sorted = sorted(D, reverse=True)
for i in D_sorted:
    CSP.append(V[D==i][0])
    
CSP_arr = np.asarray(CSP)

X_train = np.inner(X_train , CSP_arr)
X_test = np.inner(X_test , CSP_arr)

#array2csv(X_train,'/Users/Abhay/Desktop/DHL/X_train.csv')
#
#array2csv(X_test,'/Users/Abhay/Desktop/DHL/X_test.csv')
#
#class2csv(y_train,'/Users/Abhay/Desktop/DHL/y_train.csv')
#
#class2csv(y_test,'/Users/Abhay/Desktop/DHL/y_test.csv')
#    

print(CSP_arr.shape)