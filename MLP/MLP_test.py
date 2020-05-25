import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import load_model
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../data')

target_file = ['T-shirt','Trouser','Pullover',
               'Dress','Coat','Sandal','Shirt',
               'Sneaker','Bag','Ankle boot']


X_test, Y_test = mnist.test.images, mnist.test.labels

Y_test = np_utils.to_categorical(Y_test,num_classes=10)

model = load_model('mlp_fashion_mnist.h5')


plt.figure()
cnt = 30
i = 1
while(i<=12):
    img = [X_test[cnt]]
    cnt = cnt + 1
    img = np.uint8(np.double(img) * 255)
    t = model.predict(img)
    result = np.argmax(t, axis=1)
    plt.subplot(3,4,i)
    i += 1
    plt.imshow(img[0].reshape(28,28),'gray')
    plt.title(target_file[result[0]])
    plt.axis('off')
plt.show()





