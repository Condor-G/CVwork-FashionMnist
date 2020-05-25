import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../data')

X_train, Y_train = mnist.train.images, mnist.train.labels
X_test, Y_test = mnist.test.images, mnist.test.labels

Y_train = np_utils.to_categorical(Y_train,num_classes=10)
Y_test = np_utils.to_categorical(Y_test,num_classes=10)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Build LeNet-5
model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(5, 5), padding='valid', input_shape=(28, 28, 1), activation='tanh')) # C1
model.add(MaxPooling2D(pool_size=(2, 2))) # S2
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='tanh')) # C3
model.add(MaxPooling2D(pool_size=(2, 2))) # S4
model.add(Flatten())
model.add(Dense(120, activation='tanh')) # C5
model.add(Dense(84, activation='tanh')) # F6
model.add(Dense(10, activation='softmax')) # output
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=500, epochs=10, verbose=1, validation_data=(X_test, Y_test))
model.save('cnn_fashion_mnist.h5')

loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print('loss:', loss)
print('accuracy:', accuracy)









