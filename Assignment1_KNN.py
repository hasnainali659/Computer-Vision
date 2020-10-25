# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:06:13 2020

@author: DELL
"""

import cv2
import os
import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))


class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]  
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]

def load_images_from_folder(folder):
    images = []
    os.chdir(folder)
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img  , (100 , 100))
        img = np.array(np.reshape(img, (10000,)))
        images.append(img)
    return images

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy
    
x_train = np.array(load_images_from_folder(r"D:\dogs-vs-cats\data"))
y_train = np.array([0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1])

x_test = np.array(load_images_from_folder(r"D:\dogs-vs-cats\test2"))
y_test = np.array([1,1,0,0,0])

k = 3
clf = KNN(k=k)
clf.fit(x_train, y_train)


predictions = clf.predict(x_test)
print("custom KNN classification accuracy", accuracy(y_test, predictions))




