import cv2
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow import keras
from mnist_reader import load_mnist
import random

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

if __name__ == '__main__':
    degree = 30

    # Load data
    x_train, y_train_i = load_mnist('data/fashion', 'train')
    x_test, y_test_i = load_mnist('data/fashion', 't10k')

    # Only use a specific portion of data (like if debugging)
    x_train, y_train_i, x_test, y_test_i = [x[:int(len(x)*PERCENT_OF_DATA)] for x in [x_train, y_train_i, x_test, y_test_i]]

    # Rotate images
    x_train, x_test = [rotate_images(data, 0, rand_amt=degree) for data in [x_train, x_test]]

    # Normalize from 0-255 to 0.0-1.0
    x_train, x_test = [x.astype('float32') for x in [x_train, x_test]]
    x_train, x_test = [x / 255 for x in [x_train, x_test]]

    # Turn labels into one-hots
    y_train, y_test = [np.zeros((len(y), 10)) for y in [y_train_i, y_test_i]]
    y_train[np.arange(len(y_train)), y_train_i] = 1.0
    y_test[np.arange(len(y_test)), y_test_i] = 1.0
    
    # Display some training images (e.g. to check rotation)
    # [display_image(x) for x in x_train[:10]]

    model = load_model(
        './weights/cnn_best_weights_base',
        custom_objects=None,
        compile=True
    )

    # Evaluate its performance
    predictions = model.predict(x_test)
    num_correct = len([i for i, pred in enumerate(predictions) if np.argmax(pred) == np.argmax(y_test[i])])
    percent_correct = num_correct / float(len(predictions))
    print('Test set prediction accuracy: {}%'.format(percent_correct * 100))