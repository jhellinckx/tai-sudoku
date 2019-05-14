import sys
import os
import random
from operator import itemgetter, attrgetter
from math import ceil

import matplotlib.image as mpli
import matplotlib.pyplot as plt
import numpy as np
import cv2

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import plot_model
from keras.utils import np_utils

import utils

img_rows = 28
img_cols = 28
train_interpolation = cv2.INTER_AREA
predict_interpolation = cv2.INTER_AREA
img_mode = cv2.IMREAD_GRAYSCALE
input_dim = img_rows * img_cols
input_shape = (img_rows, img_cols, 1) # 2D for conv model
num_classes = 9
validation_set_proportion = 0.2

simple_model = 'base_25x9.h5'
conv_model = 'conv.h5'

DEFAULT_MODEL = conv_model

def build_simple_ocr_model():
    model = Sequential()
    model.add(Dense(units=25, activation='relu', input_dim=input_dim))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def build_conv_ocr_model():
    # https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), 
                     activation='relu', 
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model

def train_ocr_model(model, save_filename=DEFAULT_MODEL, epochs=50):
    X_train, y_train, X_val, y_val = get_char74k_dataset()
    X_train = model_reshape_dataset(X_train, model)
    X_val = model_reshape_dataset(X_val, model)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), 
              epochs=epochs, batch_size=200, verbose=1)
    model.save(save_filename)

def predict_digits(images, model_filename=DEFAULT_MODEL):
    preds = []
    model = load_model(model_filename)
    for image in images:
        if image is None:
            preds.append(0)
            continue
        preds.append(predict_digit(image, model))
    return preds

def predict_digit(image, model):
    image = cv2.resize(image, (img_rows, img_cols), 
                       interpolation=predict_interpolation)
    image = model_reshape(image, model)
    pred = model.predict(np.array([image]), batch_size=1)[0]
    return np.argmax(pred) + 1

def display_reshape(X):
    return X.reshape(X.shape[0], img_rows, img_cols)

def model_reshape(x, model):
    input_shape = model.input_shape
    if len(input_shape) == 2:
        return x.reshape(img_rows * img_cols)
    else:
        return x.reshape(img_rows, img_cols, 1)

def model_reshape_dataset(X, model):
    input_shape = model.input_shape
    if len(input_shape) == 2:
        return X.reshape(X.shape[0], img_rows * img_cols)
    else:
        return X.reshape(X.shape[0], img_rows, img_cols, 1)

def show_random_predictions(model_filename=DEFAULT_MODEL, num_predictions=20):
    X_train, y_train, X_val, y_val = get_char74k_dataset()
    model = load_model(model_filename)
    X_pred = X_val[:num_predictions]
    X_pred = model_reshape_dataset(X_pred, model)
    preds = model.predict(X_pred, batch_size=num_predictions)
    X_pred *= 255 # Rescale to 0-255 range
    X_pred = display_reshape(X_pred)
    titles = []
    for i in range(num_predictions):
        pred = preds[i]
        label = np.argmax(pred) + 1
        confidence = f'{round(pred[label - 1] * 100, 2)}'
        titles.append(f'This is a {label} ({confidence}%)')
    utils.plot_images(X_pred, titles, cols=5, figsize=(10, 10), fontsize=10)

def show_processed_image(img_array):
    img_array *= 255
    plt.figure()
    plt.imshow(img_array.reshape(img_rows, img_cols), cmap=plt.cm.gray)
    plt.show()

def get_char74k_dataset(dataset_dir='res/ocr'):
    def process_image(image):
        return cv2.resize(cv2.imread(image, img_mode), (img_rows, img_cols), 
                          interpolation=train_interpolation)
    def read_images(images):
        y = np.array(list(map(itemgetter(1), images)))
        X = np.array(list(map(process_image, map(itemgetter(0), images))))
        X = X.reshape(X.shape[0], img_rows, img_cols).astype('float32')
        X = abs(X - 255) / 255
        y = np_utils.to_categorical(y)
        return (X, y)
    train_data = []
    val_data = []
    classes_dir = f'{dataset_dir}/English/Fnt'
    for label in range(num_classes):
        dir_index = label + 2
        class_dir = f'{classes_dir}/Sample{dir_index:03d}'
        class_data = [(f'{class_dir}/{i}', label) for i in os.listdir(class_dir)]
        random.shuffle(class_data)
        slice_index = int(len(class_data) * validation_set_proportion)
        val_data += class_data[:slice_index]
        train_data += class_data[slice_index:]
    random.shuffle(train_data)
    random.shuffle(val_data)
    X_train, y_train = read_images(train_data)
    X_val, y_val = read_images(val_data)
    #show_processed_image(X_train[0])
    return (X_train, y_train, X_val, y_val)

if __name__ == '__main__':
    random.seed(1)
    #show_random_predictions(conv_model)
    model = build_conv_ocr_model()
    train_ocr_model(model, conv_model, epochs=20)
    




