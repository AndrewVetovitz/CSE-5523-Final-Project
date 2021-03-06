import cv2
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
from mnist_reader import load_mnist
import random

import matplotlib.pyplot as plt

WEIGHTS_FILENAME = 'weights/cnn_best_weights_{}'
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

def plot_accuracy(history, degree):
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('plots/cnn_train_accuracy_{}_degrees.png'.format(degree))
    plt.close()

def plot_loss(history, degree):
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('plots/cnn_train_loss_{}_degrees.png'.format(degree))
    plt.close()

if __name__ == '__main__':
    degree = 30

    # Load data
    x_train, y_train_i = load_mnist('data/fashion', 'train')
    x_test, y_test_i = load_mnist('data/fashion', 't10k')

    # Only use a specific portion of data (like if debugging)
    x_train, y_train_i, x_test, y_test_i = [x[:int(len(x)*PERCENT_OF_DATA)] for x in [x_train, y_train_i, x_test, y_test_i]]

    # Rotate images
    x_train_0, x_test_0 = [rotate_images(data, 0, 0) for data in [x_train, x_test]]
    x_train_10, x_test_10 = [rotate_images(data, 0, 10) for data in [x_train, x_test]]
    x_train_30, x_test_30 = [rotate_images(data, 0, 30) for data in [x_train, x_test]]

    # Turn labels into one-hots
    y_train, y_test = [np.zeros((len(y), 10)) for y in [y_train_i, y_test_i]]
    y_train[np.arange(len(y_train)), y_train_i] = 1.0
    y_test[np.arange(len(y_test)), y_test_i] = 1.0

    # Display some training images (e.g. to check rotation)
    # [display_image(x) for x in x_train[:10]]

    # Train the model, saving only the best weights
    checkpointer = ModelCheckpoint(WEIGHTS_FILENAME.format(degree), verbose=1, save_best_only=True)
    model = get_model()
    history = model.fit(x_train_30, y_train,
              batch_size=BATCH_SIZE,
              epochs=NUM_EPOCHS,
              verbose=1,
              validation_data=(x_test_30, y_test),
              callbacks=[checkpointer]
              )

    plot_accuracy(history, degree)
    plot_loss(history, degree)

    # Evaluate its performance 0 degrees
    predictions = model.predict(x_test_0)
    num_correct = len([i for i, pred in enumerate(predictions) if np.argmax(pred) == np.argmax(y_test[i])])
    percent_correct = num_correct / float(len(predictions))
    print('Test set 0 degree prediction accuracy: {}%'.format(percent_correct * 100))

    # Evaluate its performance 10 degrees
    predictions = model.predict(x_test_10)
    num_correct = len([i for i, pred in enumerate(predictions) if np.argmax(pred) == np.argmax(y_test[i])])
    percent_correct = num_correct / float(len(predictions))
    print('Test set 10 degree prediction accuracy: {}%'.format(percent_correct * 100))

    # Evaluate its performance 30 degrees
    predictions = model.predict(x_test_30)
    num_correct = len([i for i, pred in enumerate(predictions) if np.argmax(pred) == np.argmax(y_test[i])])
    percent_correct = num_correct / float(len(predictions))
    print('Test set 30 degree prediction accuracy: {}%'.format(percent_correct * 100))