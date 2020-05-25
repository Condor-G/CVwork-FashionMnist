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

# Build MLP
model = Sequential()

model.add(Dense(units=256,
		input_dim=784,
		kernel_initializer='normal',
		activation='relu'))

model.add(Dense(units=10,
		kernel_initializer='normal',
		activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=500, epochs=10, verbose=1, validation_data=(X_test, Y_test))
model.save('mlp_fashion_mnist.h5')

loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print('loss:', loss)
print('accuracy:', accuracy)









