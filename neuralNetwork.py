# Let NeuralNetwork deal with details like bias.
import numpy as np
import networkComponents
import activations

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

        self.V = Model((self.n_hid, self.n_in + 1))
        self.W = Model((self.n_out, self.n_hid + 1))


    def forward_pass(self, x):
        """
        Args:
            x -- Single data point with x.shape == n_in
        """
        raise NotImplementedError

    def backward_pass(self):
        raise NotImplementedError



