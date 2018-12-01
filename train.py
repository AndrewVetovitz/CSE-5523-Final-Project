'''Trains on the Fashion MNIST dataset.

Still have to figure out which training method we'll use
and whether we'll use a library or not.

Kind of think that with how easy MNIST is we might want to do the
math ourselves if we can, but either way works.'''

from mnist_reader import load_mnist

if __name__ == '__main__':
    x_train, y_train = load_mnist('data/fashion', 'train')
    x_test, y_test = load_mnist('data/fashion', 't10k')