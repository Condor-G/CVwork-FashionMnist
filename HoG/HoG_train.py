import warnings
warnings.filterwarnings("ignore")

import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from skimage.feature import hog
from scipy.cluster.vq import vq
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data
import time

print(time.asctime( time.localtime(time.time()) ))
start_time = time.time()

mnist = input_data.read_data_sets('../data')
x_train = mnist.train.images
y_train = mnist.train.labels


#limit = 5000 # 最大训练个数
limit = y_train.size

data = [] # HoG特征
label = []
for i in range(limit):
    img = x_train[i].reshape(28,28)
    img = np.uint8(np.double(img) * 255)
    fd = hog(img)
    data.append(fd)
    label.append(y_train[i])

data = np.array(data)
label = np.array(label)

x_tra, x_val, y_tra, y_val = train_test_split(data,label,test_size=0.2)

print('train file:',y_tra.size)
print('val file:',y_val.size)

# 构建线性SVM对象并训练
clf = LinearSVC(C=1, loss="hinge").fit(x_tra, y_tra)
# 训练数据预测正确率
print ('accuracy:',clf.score(x_val, y_val))


end_time = time.time()
print("Execution Time: ", int(end_time - start_time),'s')

# save the training model as pickle
with open('hog.pickle','wb') as fw:
    pickle.dump(clf,fw)
print('Trainning successfully and save the model')
