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
import hdbscan
from sklearn.cluster import DBSCAN

COLORS = [[255,0,0],
          [0,255,0],
          [0,0,255],
          [255,255,0], #yello
          [0,255,255], #Cyan
          [255,0,255], #Magenta
          [192,192,192], #Silver
          [128,128,128], #Gray
          [128,0,0], #Maroon
          [128,128,0], #Olive
          [0,128,0], #Green
          [128,0,128], #Purple
          [0,128,128], #Teal
          [0,0,128]] #Navy

#%% Read Image
image = cv2.imread('C:\\Users\\Piyush\\Pictures\\Deadpool_Movie.jpg')
image = cv2.bilateralFilter(image,15,75,75)
h,w,c = image.shape
image = cv2.resize(image, (96,96))
image = cv2.bilateralFilter(image,15,75,75)

h,w,c = image.shape
print(image.shape)

#%% Reshape Image for clustering
X = np.reshape(image, (h*w, c))
print(X.shape)

#%% HDBSCAN
scanner = hdbscan.HDBSCAN()
print("Computing Clusters")
start = time.time()

scanner.fit(X)

print("Done Computing")
print("time Taken:",time.time() - start)

n_clusters = len(scanner.labels_)
print("Cluster Size:",n_clusters)


unique, counts = np.unique(scanner.labels_, return_counts=True)
unique = np.reshape(unique, (-1,1))
counts = np.reshape(counts, (-1,1))

unique = np.hstack((unique, counts))
print(unique.shape)
sorted_list = unique[np.argsort(unique[:, 1])[::-1]]
print(sorted_list[0:10])

#%% DBSCAN
print("Computing Clusters")
start = time.time()
clustering = DBSCAN(eps=3, min_samples=2).fit(X)
print("Done Computing")
print("time Taken:",time.time() - start)


n_clusters = len(scanner.labels_)
print("Cluster Size:",n_clusters)

unique, counts = np.unique(clustering.labels_, return_counts=True)
unique = np.reshape(unique, (-1,1))
counts = np.reshape(counts, (-1,1))

unique = np.hstack((unique, counts))
print(unique.shape)
sorted_list = unique[np.argsort(unique[:, 1])[::-1]]
print(sorted_list[0:10])
#%% Display Labels HDBSCAN
print(scanner.labels_)
original = np.reshape(scanner.labels_, (h,w))
print(original.shape)

copy_HDBSCAN = image.copy()
for i in range(10):
    copy_HDBSCAN[original == sorted_list[i][0]] = COLORS[i]

#%% Display Labels DBSCAN

print(clustering.labels_)
original = np.reshape(clustering.labels_, (h,w))
print(original.shape)

copy_DBSCAN = image.copy()
for i in range(10):
    copy_DBSCAN[original == sorted_list[i][0]] = COLORS[i]
    
#%% Display Picture
cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.namedWindow("Image_HDBSCAN", cv2.WINDOW_NORMAL)
cv2.namedWindow("Image_DBSCAN", cv2.WINDOW_NORMAL)
cv2.imshow("Original", image)
cv2.imshow("Image_HDBSCAN", copy_HDBSCAN)
cv2.imshow("Image_DBSCAN", copy_DBSCAN)
cv2.waitKey(0)
cv2.destroyAllWindows()

