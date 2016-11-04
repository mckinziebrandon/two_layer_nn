#!/usr/bin/python
import sys, getopt

import sklearn.metrics as metrics
from mnist import MNIST
import numpy as np
import pdb
import matplotlib.pyplot as plt

import util
from neuralNetwork import NeuralNetwork

"""
Change this code however you want.
"""

n_save = 60000

def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    # The test labels are meaningless,
    # since you're replacing the official MNIST test set with our own test set
    X_val, labels_val = map(np.array, mndata.load_testing())
    # Remember to center and normalize the data...
    return X_train, labels_train, X_val, labels_val

def get_data(reload=False):
    if reload:
        X_train, labels_train, X_val, labels_val = load_dataset()
        np.savez('less_data_{0}.npz'.format(n_save),
                X_train      = X_train[:n_save],
                labels_train = labels_train[:n_save],
                X_val        = X_val[:n_save],
                labels_val   = labels_val[:n_save])
    else:
        files        = np.load('less_data_{0}.npz'.format(n_save))
        X_train      = files['X_train']
        labels_train = files['labels_train']
        X_val        = files['X_val']
        labels_val   = files['labels_val']
    return X_train, labels_train, X_val, labels_val


def plot_error(x_axis, y_axis, title):
    plt.style.use('ggplot')

    fig = plt.figure()
    fig.suptitle("Stochastic Gradient Descent", fontsize='x-large')

    ax1 = fig.add_subplot(111)
    ax1.plot(x_axis, y_axis)
    ax1.set_xlabel('Number of Iterations')
    ax1.set_ylabel('Training Error')
    plt.text(0.85 * x_axis[-1], 0.85, r'$\alpha=5e-2,\ \lambda=0.01$', fontsize='x-large')

    #fig.savefig("{0}.png".format(title))
    #fig.savefig("{0}.pdf".format(title))
    plt.show()


def trainNeuralNetwork(reload=False):

    # Get the MNIST data, either by loading saved data or getting it new.
    X_train, labels_train, X_test, labels_test = get_data(reload)

    print("X_train.shape", X_train.shape)
    print("labels_train.shape", labels_train.shape)
    #print("X_val.shape", X_val.shape)
    #print("labels_val.shape", labels_val.shape)

    # ___________ Initialize the neural network. ____________
    neural_net = NeuralNetwork( n_in=X_train.shape[1],
                                n_hid=200,
                                n_out=10,
                                eta=1e-3,
                                decay_const=0.7,
                                l2=0.01,
                                n_epochs=1)

    neural_net.set_data(X_train, labels_train)


    train_err = []
    n_train = int(50e3)
    batches = np.array_split(np.arange(n_train), 5000)

    epochs  = np.arange(neural_net.n_epochs)
    x_range = len(epochs) * len(batches)
    x_axis  = np.arange(0, x_range, x_range//50)

    for i in epochs:
        print('========== EPOCH {} ======'.format(i))
        neural_net.new_epoch(i)
        for j, batch in enumerate(batches):

            # Tell neural net what data to train with.
            neural_net.set_active(batch)

            # Calculate values along feedforward.
            X, S_h, H, S_o, O = neural_net.forward_pass()

            # Get the training loss at this iteration.
            if i * len(batches) + j in x_axis:
                train_err.append(1.0 - neural_net.train_accuracy())

            # Update weights via backprop.
            neural_net.train(X, H, O)


    # Save Kaggle predictions in CSV file.
    if False:
        pred_labels_test = neural_net.predict_test(X_test)
        Id          = np.reshape(np.arange(1, 1+X_test.shape[0]), (X_test.shape[0], 1))
        Category    = np.reshape(pred_labels_test, (X_test.shape[0], 1))
        columns     = np.hstack((Id, Category))
        np.savetxt('predictions.csv', columns, delimiter=',', header='Id, Category', fmt='%d')

    neural_net.print_results()
    plot_error(x_axis, train_err, "something")
    #pdb.set_trace()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'reload':
        trainNeuralNetwork(reload=True)
    else:
        trainNeuralNetwork()
    print("I am a robot. Bleep Bloop.")


