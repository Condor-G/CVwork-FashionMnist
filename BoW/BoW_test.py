import warnings
warnings.filterwarnings("ignore")

import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq
from sklearn.svm import LinearSVC
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../data')
x_test = mnist.test.images
y_test = mnist.test.labels

with open('bow_kmeans.pickle','rb') as fr:
    k_means = pickle.load(fr)
with open('bow_clf.pickle','rb') as fr:
    clf = pickle.load(fr)
with open('bow_count.pickle','rb') as fr:
    count = pickle.load(fr)

target_file = ['T-shirt','Trouser','Pullover',
               'Dress','Coat','Sandal','Shirt',
               'Sneaker','Bag','Ankle boot']

plt.figure()
cnt = 30
i = 1
while(i<=12):
    img = x_test[cnt].reshape(28,28)
    cnt = cnt + 1
    img = np.uint8(np.double(img) * 255)
    sift = cv2.xfeatures2d.SIFT_create()
    kp,des = sift.detectAndCompute(img,None)
    if des is None:
        continue
    words, distance = vq(des, k_means.cluster_centers_)
    image_features_search = np.zeros((int(count)), "float32")
    for w in words:
        image_features_search[w] += 1
    t = clf.predict(image_features_search.reshape(1,-1))
    plt.subplot(3,4,i)
    i += 1
    plt.imshow(img,'gray')
    plt.title(target_file[t[0]])
    plt.axis('off')
plt.show()

