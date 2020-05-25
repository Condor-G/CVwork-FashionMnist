import warnings
warnings.filterwarnings("ignore")

import os
import cv2
import pickle
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt
from imutils import paths
from scipy.cluster.vq import vq
from sklearn.svm import LinearSVC
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../data')
x_test = mnist.test.images
y_test = mnist.test.labels

with open('hog.pickle','rb') as fr:
    clf = pickle.load(fr)

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
    fd = hog(img)
    t = clf.predict([fd])
    plt.subplot(3,4,i)
    i += 1
    plt.imshow(img,'gray')
    plt.title(target_file[t[0]])
    plt.axis('off')
plt.show()

