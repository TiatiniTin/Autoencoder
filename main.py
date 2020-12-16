from IPython.core.pylabtools import figsize
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import matplotlib

from tensorflow import keras

counter = 0
data_input = []
data_output = []

for k in range(3):
    fileName = k + 1

    path_template_input = 'Dataset_mini/Sentinel/{file}.tif'
    path_template_output = 'Dataset_mini/Jaxa/{file}.tif'

    path_input = path_template_input.format(file=fileName)
    path_output = path_template_output.format(file=fileName)

    # загружаем изображение, сглаживаем его в 30x30x3=2700 пикселей и
    # добавляем в список
    image = cv2.imread(path_input).flatten()
    data_input.append(image)

    image = cv2.imread(path_output).flatten()
    data_output.append(image)

# масштабируем интенсивности пикселей в диапазон [0, 1]
# data_input = np.array(data_input, dtype="float") / 255.0
# data_output = np.array(data_output, dtype="float") / 255.0

# разбиваем данные на обучающую и тестовую выборки, используя 75%
# данных для обучения и оставшиеся 25% для тестирования
(trainX, testX, trainY, testY) = train_test_split(data_input,
    data_output, test_size=0.25, random_state=42)



encoder = keras.models.Sequential([
    keras.layers.Reshape([30, 30, 3], input_shape=[30, 30, 3]),
    keras.layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
    keras.layers.MaxPool2D(pool_size=2)
])

#encoder.predict(testX[0].reshape((1, 30, 30))).shape
#encoder.predict(testX[0].reshape((1, 30, 30))).shape

decoder = keras.models.Sequential([
    keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=2, padding="valid",
                                 activation="relu",
                                 input_shape=[3, 3, 64]),
    keras.layers.Conv2DTranspose(16, kernel_size=(3, 3), strides=2, padding="same",
                                 activation="relu"),
    keras.layers.Conv2DTranspose(3, kernel_size=(3, 3), strides=2, padding="same",
                                 activation="sigmoid"),
    keras.layers.Reshape([30, 30, 3])
])

stacked_autoencoder = keras.models.Sequential([encoder, decoder])

stacked_autoencoder.compile(loss="binary_crossentropy",
                            optimizer='adam')

history = stacked_autoencoder.fit(trainX, trainY, epochs=10,
                         validation_data=[testX, testY])

figsize(20, 5)
for i in range(8):
    plt.subplot(2, 8, i + 1)
    pred = stacked_autoencoder.predict(testX[i].reshape((1, 30, 30)))
    plt.imshow(testX[i])

    plt.subplot(2, 8, i + 8 + 1)
    plt.imshow(pred.reshape((30, 30)))