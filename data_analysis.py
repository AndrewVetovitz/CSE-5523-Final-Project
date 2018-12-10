import cv2
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
from mnist_reader import load_mnist
import random

WEIGHTS_FILENAME = 'weights/cnn_best_weights'
NUM_EPOCHS = 5
BATCH_SIZE = 64

# Percent of data to train & test on (useful to decrease for debugging)
PERCENT_OF_DATA = 1.0

def display_image(image, wait_time=0):
    """Displays an image until and waits the given amount of
    time for a key press before continuing."""
    cv2.imshow('', image)
    cv2.waitKey(wait_time)

def rotate_image(image, angle, rand_amt=0):
    """Rotate an image by to the specified angle. Note that an
    incremented angle rotates counter-clockwise.

    rand(-rand_amt, rand_amt) is added to the angle before
    rotation."""
    image = np.reshape(image, (28,28))
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    angle += random.randint(-rand_amt, rand_amt)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    # Convert to BGR because the Conv2D layers don't like grayscale
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

def rotate_images(images, angle, rand_amt=0):
    return np.array([rotate_image(im, angle, rand_amt) for im in images])

def get_model():
    """Returns the CNN model to be used for training."""
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28,28,3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    degree = 30

    # Load data
    x_train, y_train_i = load_mnist('data/fashion', 'train')
    x_test, y_test_i = load_mnist('data/fashion', 't10k')

    result = [0] * 10

    for y in y_train_i:
        result[y] += 1

    print(result)