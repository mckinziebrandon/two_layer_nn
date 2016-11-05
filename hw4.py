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

def get_data(reload=False, wantDeskew=True):
    if reload:
        X_train, labels_train, X_test, labels_test = load_dataset()
        X_train = X_train/255.
        X_test = X_test/255.
        if wantDeskew:
            print("Deskewing...")
            X_train = util.deskewAll(X_train)
            X_test = util.deskewAll(X_test)
        np.savez('less_data_{0}.npz'.format(n_save),
                X_train      = X_train[:n_save],
                labels_train = labels_train[:n_save],
                X_test        = X_test[:n_save],
                labels_test   = labels_test[:n_save])
    else:
        files        = np.load('less_data_{0}.npz'.format(n_save))
        X_train      = files['X_train']
        labels_train = files['labels_train']
        X_test        = files['X_test']
        labels_test   = files['labels_test']
    return X_train, labels_train, X_test, labels_test

def trainNeuralNetwork(reload=False, wantDeskew=True):

    # Get the MNIST data, either by loading saved data or getting it new.
    X_train, labels_train, X_test, labels_test = get_data(reload, wantDeskew)
    #labels_train = util.deskew(labels_train)
    #labels_test = util.deskew(labels_test)

    print("X_train.shape", X_train.shape)
    print("labels_train.shape", labels_train.shape)

    # ___________ Initialize the neural network. ____________
    neural_net = NeuralNetwork( n_in=X_train.shape[1],
                                n_hid=1200,
                                n_out=10,
                                eta=0.1,
                                decay_const=0.8,
                                alpha=0.9,
                                l2=0.07,
                                batch_size=50,
                                n_epochs=15)

    neural_net.set_data(X_train, labels_train)

    accuracy = []
    loss = []

    n_train = int(50e3)
    n_batches = n_train//neural_net.batch_size
    batches = np.array_split(np.arange(n_train), n_batches)
    print("Splitting into", len(batches), "of size", neural_net.batch_size)

    epochs  = np.arange(neural_net.n_epochs)
    n_iter_total = len(epochs) * len(batches)
    x_axis  = np.arange(0, n_iter_total, n_iter_total//100)
    print("preparing to collect", len(x_axis), "points to plot.")

    print("Beginning", n_iter_total,"iterations of minibatch gradient descent.")
    for i in epochs:
        print('\n========== EPOCH {} ======'.format(i))
        neural_net.new_epoch(i)
        for j, batch in enumerate(batches):

            # Tell neural net what data to train with.
            neural_net.set_active(batch)

            # Calculate values along feedforward.
            X, S_h, H, S_o, O = neural_net.forward_pass()

            # Get the training loss at this iteration.
            if i * len(batches) + j in x_axis:
                print(".", end=" "); sys.stdout.flush()
                loss.append(neural_net.get_loss())
                accuracy.append(neural_net.train_accuracy())

            # Update weights via backprop.
            neural_net.train(X, H, O)


    # Save Kaggle predictions in CSV file.
    if True:
        pred_labels_test = neural_net.predict_test(util.preprocess(X_test))
        Id          = np.reshape(np.arange(1, 1+X_test.shape[0]), (X_test.shape[0], 1))
        Category    = np.reshape(pred_labels_test, (X_test.shape[0], 1))
        columns     = np.hstack((Id, Category))
        np.savetxt('predictions.csv', columns, delimiter=',', header='Id, Category', fmt='%d')

    neural_net.print_results()
    util.plot_error(x_axis, loss, accuracy, neural_net.get_params())
    #pdb.set_trace()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'reload':
        trainNeuralNetwork(reload=True)
    else:
        trainNeuralNetwork()
    print("I am a robot. Bleep Bloop.")


