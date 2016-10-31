# Let NeuralNetwork deal with details like bias.
import numpy as np
from networkComponents import *
import activations
import util

class NeuralNetwork(object):

    # Class attributes here. Shared among all instances.

    def __init__(self, n_in, n_hid, n_out):
        """
        Instance Variables:
            self.n_in   number of inputs (features) to network.
            self.n_hid  number of hidden units in single hidden layer.
            self.n_out  number of output neurons.
        """
        self.n_in   = n_in
        self.n_hid  = n_hid
        self.n_out  = n_out

        # Initialize values of all units.
        self.x = np.zeros(self.n_in)
        self.h = np.zeros(self.n_hid)
        self.o = np.zeros(self.n_out)

        #self.V = Model((self.n_hid, self.n_in + 1))
        #self.W = Model((self.n_out, self.n_hid + 1))

        self.V = np.random.randn(self.n_hid, self.n_in + 1)
        self.W = np.random.randn(self.n_out, self.n_hid + 1)


    def forward_pass(self, x, y):
        """
        Computes the values of all units (neurons) given some input data point x.

        Args:
            x -- Single data point with x.shape == (n_in,)
            y -- Corresponding one-hot encoded truth labels with shape (n_out,)

        Returns: the cross-entropy loss for this point x.
        """
        self.x = x
        x = np.append(1, x)

        # First step: Compute values of hidden neuron units.
        S_h = np.dot(self.V, x)
        self.h   = activations.relu(S_h)

        # Second step: Compute values of output units.
        h = np.append(1, self.h)
        S_o = np.dot(self.W, h)
        self.o   = activations.softmax(S_o)

        return util.cross_entropy(self.o, y)



    def backward_pass(self):
        raise NotImplementedError



