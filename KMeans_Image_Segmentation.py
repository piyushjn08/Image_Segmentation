# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 18:51:00 2019

@author: Piyush
"""
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import cv2
import time
import numpy as np

image = cv2.imread('C:\\Users\\Piyush\\Pictures\\Deadpool_Movie.jpg')
image = cv2.bilateralFilter(image,15,75,75)
h,w,c = image.shape
image = cv2.resize(image, (256,256))
image = cv2.bilateralFilter(image,15,75,75)
h,w,c = image.shape
copy = image.copy()
print(image.shape)
X = np.reshape(image, (h*w, c))
print(X.shape)

COLORS = [[255,0,0],
          [0,255,0],
          [0,0,255]]
#%% K-Means
print("Computing K-Means")
start = time.time()
kmeans = MiniBatchKMeans(n_clusters=3, random_state=0, batch_size=10000).fit(X)
print("Done Computing")
print("time Taken:",time.time() - start)

#%% Display Labels
print(kmeans.labels_)
original = np.reshape(kmeans.labels_, (h,w))
print(original.shape)

copy[ original == 0] = COLORS[0]
copy[ original == 1] = COLORS[1]
copy[ original == 2] = COLORS[2]

#%% Display Picture
cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow("Original", image)
cv2.imshow("Image", copy)
cv2.waitKey(0)
cv2.destroyAllWindows()

