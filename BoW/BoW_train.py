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
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data
import time

print(time.asctime( time.localtime(time.time()) ))
start_time = time.time()

mnist = input_data.read_data_sets('../data')
x_train = mnist.train.images
y_train = mnist.train.labels

sifts_img = [] # 存放所有图像的文件名和sift特征

limit = 10000 # 最大训练个数
#limit = y_train.size

count = 0 # 词袋特征个数
num = 0 # 有效个数
label = []
for i in range(limit):
    img = x_train[i].reshape(28,28)
    img = np.uint8(np.double(img) * 255)
    sift = cv2.xfeatures2d.SIFT_create()
    kp,des = sift.detectAndCompute(img,None)
    if des is None:
        continue
    sifts_img.append(des)
    label.append(y_train[i])
    count = count + des.shape[0]
    num = num + 1

label = np.array(label)

data = sifts_img[0]
for des in sifts_img[1:]:
    data = np.vstack((data, des))

print("train file:",num)
count = int(count / 40)
count = max(4,count)

# 对sift特征进行聚类
k_means = KMeans(n_clusters=int(count), n_init=4)
k_means.fit(data)


# 构建所有样本的词袋表示
image_features = np.zeros([int(num),int(count)],'float32')
for i in range(int(num)):
    ws, d = vq(sifts_img[i],k_means.cluster_centers_)# 计算各个sift特征所属的视觉词汇
    for w in ws:
        image_features[i][w] += 1  # 对应视觉词汇位置元素加1


x_tra, x_val, y_tra, y_val = train_test_split(image_features,label,test_size=0.2)
# 构建线性SVM对象并训练
clf = LinearSVC(C=1, loss="hinge").fit(x_tra, y_tra)
# 训练数据预测正确率
print (clf.score(x_val, y_val))


end_time = time.time()
print("Execution Time: ", int(end_time - start_time),'s')

# save the training model as pickle
with open('bow_kmeans.pickle','wb') as fw:
    pickle.dump(k_means,fw)
with open('bow_clf.pickle','wb') as fw:
    pickle.dump(clf,fw)
with open('bow_count.pickle','wb') as fw:
    pickle.dump(count,fw)
print('Trainning successfully and save the model')
