# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix

data = pd.read_csv('R:\\ML\\Project 1\\wdbc.dataset', header = None)
data = data.drop(columns = 0)
X = data.iloc[:, 1:31].values
Y = data.iloc[:, 0].values
X = preprocessing.normalize(X)
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
##X_valid, X_test, Y_valid, Y_test = train_test_split(X_test1, Y_test1, test_size = 0.5, random_state = 0)
X_train, Y_train = X_train.T, Y_train.T
epochs = 1000
alpha = 0.25
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def predict(w1, b1, x):
        m1 = x.shape[0]
        Y_prediction = np.zeros((1,m1))
        A = sigmoid(np.dot(w1.T, x.T) + b1)
        for i in range(A.shape[1]):
            Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
        return Y_prediction.T
losstrack = []
m = X_train.shape[0]
w = np.zeros((X_train.shape[0],1))
b = 0
for epoch in range(epochs):
    z = np.dot(w.T, X_train) + b
    p = sigmoid(z)
    cost = -np.sum(np.multiply(np.log(p), Y_train) + np.multiply((1 - Y_train), np.log(1 - p)))/m
    losstrack.append(np.squeeze(cost))
    dz = p - Y_train
    dw = (1 / m) * np.dot(X_train, dz.T)
    db = (1 / m) * np.sum(dz)
    w = w - alpha * dw
    b = b - alpha * db
    Y_prediction_test = predict(w, b, X_test)
    conf1 = confusion_matrix(Y_test, Y_prediction_test)
    plt.plot(losstrack)
    
   
Y_prediction_test = predict(w, b, X_test)
conf = confusion_matrix(Y_test, Y_prediction_test)

TP = conf[0][0]
FN = conf[0][1]
FP = conf[1][0]
TN = conf[1][1]

accuracy = (TP+TN)/(TP+TN+FP+FN)
print("Accuracy:" +str(accuracy))

precision = (TP)/(TP+FP)
print("Precision:" +str(precision))

recall = (TP)/(TP+FN)
print("Recall:" +str(recall))
