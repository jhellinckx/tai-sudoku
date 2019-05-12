import sys
import os
import random
from operator import itemgetter, attrgetter

import matplotlib.image as mpli
import matplotlib.pyplot as plt
import numpy as np
import cv2

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
from keras.utils import np_utils

img_rows = 28
img_cols = 28
train_interpolation = cv2.INTER_AREA
img_mode = cv2.IMREAD_GRAYSCALE
input_dim = img_rows * img_cols
num_classes = 9
validation_set_proportion = 0.2

def build_ocr_model():
    model = Sequential()
    model.add(Dense(units=25, activation='relu', input_dim=input_dim))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def train_ocr_model():
    X_train, y_train, X_val, y_val = get_char74k_dataset()
    model = build_ocr_model()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=200, verbose=1)
    model_name = '25x9'
    model.save(f'{model_name}.h5')


def show_processed_image(img_array):
    img_array *= 255
    plt.figure()
    plt.imshow(img_array.reshape(img_rows, img_cols), cmap=plt.cm.gray)
    plt.show()

def get_char74k_dataset(dataset_dir='res/ocr'):
    def process_image(image):
        return cv2.resize(cv2.imread(image, img_mode), (img_rows, img_cols), interpolation=train_interpolation)
    def read_images(images):
        y = np.array(list(map(itemgetter(1), images)))
        X = np.array(list(map(process_image, map(itemgetter(0), images))))
        X = X.reshape(X.shape[0], img_rows * img_cols).astype('float32')
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
    X_train, y_train = read_images(train_data)

    X_val, y_val = read_images(val_data)
    
    return (X_train, y_train, X_val, y_val)
    

if __name__ == '__main__':
    train_ocr_model()




