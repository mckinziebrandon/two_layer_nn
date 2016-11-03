#!/usr/bin/python
import sys, getopt

from mnist import MNIST
import numpy as np
import pdb

import util
from neuralNetwork import NeuralNetwork

"""
Change this code however you want.
"""

n_save = 50

def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    # The test labels are meaningless,
    # since you're replacing the official MNIST test set with our own test set
    X_val, labels_val = map(np.array, mndata.load_testing())
    # Remember to center and normalize the data...
    return X_train, labels_train, X_val, labels_val


def trainNeuralNetwork(reload=False):

    # Get the MNIST data.
    if reload:
        X_train, labels_train, X_val, labels_val = load_dataset()
        np.savez('less_data_{0}.npz'.format(n_save),
                X_train = X_train[:n_save],
                labels_train=labels_train[:n_save],
                X_val=X_val[:n_save],
                labels_val=labels_val[:n_save])
    else:
        files = np.load('less_data_{0}.npz'.format(n_save))
        X_train = files['X_train']
        labels_train = files['labels_train']
        X_val = files['X_val']
        labels_val = files['labels_val']


    #pdb.set_trace()

    # Number of inputs neurons = number of features (usually denoted as 'd')
    n_in = X_train.shape[1]
    # Number of hidden neurons (arbitrary).
    n_hid = 200
    # Number of outpus = number of classes = number of unique single digits.
    n_out = 10

    Y_train     = util.one_hot(labels_train, n_out)
    Y_val       = util.one_hot(labels_val, n_out)

    print("X_train.shape", X_train.shape)
    print("labels_train.shape", labels_train.shape)
    print("X_val.shape", X_val.shape)
    print("labels_val.shape", labels_val.shape)

    # ___________ Initialize the neural network. ____________
    neural_net = NeuralNetwork(n_in, n_hid, n_out)
    neural_net.set_data(X_train, labels_train)

    X, S_h, H, S_o, O = neural_net.forward_pass(verbose=True, debug=False)
    do, dh= neural_net.get_deltas(O, H)
    cost = neural_net.get_cost(O)
    neural_net.train()
    pdb.set_trace()


    batches = [np.arange(a, a+4) for a in [0, 10, 20, 30, 40]]

    cost = []
    for batch in batches:
        # Tell neural net what data to train with.
        neural_net.set_active(batch)
        # Calculate values along feedforward.
        X, S_h, H, S_o, O = self.forward_pass()
        # Get the training loss at this iteration.
        cost.append(neural_net.get_cost(O))
        # Update weights via backprop.
        neural_net.train(X, H, O)



    #num_samples = np.arange(neural_net.data['X'].shape[0])
    #np.random.shuffle(num_samples)
    #for i in num_samples:
        # Calculate the deltas back for input X[i].
        #neural_net.backward_pass(i)
        #neural_net.update_weights(i)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'reload':
        trainNeuralNetwork(reload=True)
    else:
        trainNeuralNetwork()
    print("I am a robot. Bleep Bloop.")


