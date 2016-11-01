from mnist import MNIST
import numpy as np

import util
from neuralNetwork import NeuralNetwork

"""
Change this code however you want.
"""

def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    # The test labels are meaningless,
    # since you're replacing the official MNIST test set with our own test set
    X_val, labels_val = map(np.array, mndata.load_testing())
    # Remember to center and normalize the data...
    return X_train, labels_train, X_val, labels_val


def trainNeuralNetwork():

    # Get the MNIST data.
    X_train, labels_train, X_val, labels_val = load_dataset()

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

    batches = [np.arange(a, a+4) for a in [0, 10, 20, 30, 40]]

    for batch in batches:
        loss = neural_net.forward_pass(batch, debug=True)
        delta_o, delta_h = neural_net.backward_pass(batch)
        neural_net.update_weights(delta_o, delta_h)


    #num_samples = np.arange(neural_net.data['X'].shape[0])
    #np.random.shuffle(num_samples)
    #for i in num_samples:
        # Calculate the deltas back for input X[i].
        #neural_net.backward_pass(i)
        #neural_net.update_weights(i)


if __name__ == "__main__":
    trainNeuralNetwork()
    print("I am a robot. Bleep Bloop.")


