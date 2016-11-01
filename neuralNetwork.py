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
        # TODO: Actually, shapes are N_data by _____ for all. Fix.
        self.X = np.zeros(self.n_in)
        self.H = np.zeros(self.n_hid)
        self.O = np.zeros(self.n_out)

        # Weights initialization.
        self.V = np.random.randn(self.n_hid, self.n_in + 1)
        self.W = np.random.randn(self.n_out, self.n_hid + 1)

    def set_data(self, X_data, Y_data):
        """ Store data in instance attribute self.data,
            which is a dictionary with keys 'X' and 'Y'. """
        self.data = self._get_shuffled_data_dict(X_data, Y_data)
        self.data['X'] = util.standardized(self.data['X'])

    def _get_shuffled_data_dict(self, X_data, Y_data):
        # First zip the data along the same axis before shuffling.
        joined_data = np.concatenate((X_data, Y_data[:, None]), axis=1)
        # Now we can shuffle the data together.
        np.random.shuffle(joined_data)
        # Separate back and return in a dictionary.
        (new_x, new_y) = joined_data[:, :self.n_in], joined_data[:, self.n_in:]
        return {'X': new_x, 'Y': new_y.flatten()}

    def forward_pass(self, debug=False):
        """
        Computes the values of all units (neurons) over ALL sample points (vectorized).

        Args:
            x -- Single data point with x.shape == (n_in,)
            y -- Corresponding one-hot encoded truth labels with shape (n_out,)

        Returns: the cross-entropy loss for this point x.
        """

        # First step: Compute values of hidden neuron units.
        if (debug): print("Forward Pass: Computing 1st layer . . . ")
        S_h     = util.withBias(self.data['X']) @ self.V.T
        self.H  = activations.relu(S_h)

        # Second step: Compute values of output units.
        if (debug): print("Forward Pass: Computing 2nd layer . . . ")
        S_o     = util.withBias(self.H) @ self.W.T
        self.O  = activations.softmax(S_o)

        return util.cross_entropy(self.O, self.data['Y'])



    def backward_pass(self):
        raise NotImplementedError



