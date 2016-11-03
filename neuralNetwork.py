# Let NeuralNetwork deal with details like bias.
import sklearn.metrics as metrics
import numpy as np
import pdb
from networkComponents import *
import activations
import util

class NeuralNetwork(object):


    def __init__(self, eta, n_in=784, n_hid=200, n_out=10):
        """
        Instance Variables:
            self.n_in   number of inputs (features) to network.
            self.n_hid  number of hidden units in single hidden layer.
            self.n_out  number of output neurons.
        """
        # Class attributes here. Shared among all instances.
        self.eta = eta

        self.n_in   = n_in
        self.n_hid  = n_hid
        self.n_out  = n_out

        # Weights initialization.
        self.V = util.weight_init((self.n_hid, self.n_in + 1),  how="uniform")
        self.W = util.weight_init((self.n_out, self.n_hid + 1), how="uniform")

    def set_data(self, X_data, Y_data):
        """ Store data in instance attribute self.data,
            which is a dictionary with keys 'X' and 'Y'. """

        self.data = self._get_shuffled_data_dict(X_data, Y_data)
        self.data['X']       = util.standardized(self.data['X'])
        self.data['one_hot'] = util.one_hot(self.data['labels'])
        self.X = self.data['X']
        self.n_data = self.X.shape[0]
        self.Y_hot = self.data['one_hot']
        self.labels = self.data['labels']

    def set_active(self, indices=None):
        if indices is None: return
        self.X = self.data['X'][indices]
        self.Y_hot = self.data['one_hot'][indices]
        self.labels = self.data['labels'][indices]
        self.n_data = self.X.shape[0]

    def _get_shuffled_data_dict(self, X_data, Y_data):
        # First zip the data along the same axis before shuffling.
        joined_data = np.concatenate((X_data, Y_data[:, None]), axis=1)
        # Now we can shuffle the data together.
        np.random.shuffle(joined_data)
        # Separate back and return in a dictionary.
        (new_x, new_y) = joined_data[:, :self.n_in], joined_data[:, self.n_in:]
        return {'X': new_x, 'labels': new_y.flatten()}

    def forward_pass(self, verbose=False, debug=False):
        """
        Computes the values of all units (neurons) over ALL sample points (vectorized).

        Args:
            x -- Single data point with x.shape == (n_in,)
            y -- Corresponding one-hot encoded truth labels with shape (n_out,)

        Returns: the cross-entropy loss for this point x.
        """
        if (verbose): print("\n _______ Forward Pass _______")

        # Zeroth step: Outputs of input units.
        X = self.X
        if (verbose): print("\t X.shape", X.shape)

        # First step: Outputs (H) of hidden units.
        if (verbose): print("\t Computing 1st layer . . . ")
        S_h     = util.withBias(X) @ self.V.T
        H       = activations.relu(S_h)

        # Second step: Outputs (O) of output units.
        if (verbose): print("\t Computing 2nd layer . . . ")
        S_o     = util.withBias(H) @ self.W.T
        O  = activations.softmax(S_o, verbose)

        if debug: pdb.set_trace()

        return X, S_h, H, S_o, O

    def get_cost(self, outputs):
        return util.cross_entropy(outputs, self.Y_hot)


    def get_deltas(self, O, H):
        """
        Returns:
            delta_o, delta_1,
        """
        from activations import relu_deriv

        delta_o = O - self.Y_hot
        delta_h = relu_deriv(H) * (delta_o @ self.W[:, 1:])
        return delta_o, delta_h


    def train(self, X, H, O):
        delta_o, delta_h = self.get_deltas(O, H)

        #pdb.set_trace()
        self.W[:, 1:] -= (self.eta/self.n_data) * delta_o.T @ H
        self.W[:, :1] -=  (self.eta/self.n_data) * delta_o.sum(axis=0)[:, None]

        self.V[:, 1:] -= (self.eta/self.n_data) * delta_h.T @ X
        self.V[:, :1] -=  (self.eta/self.n_data) * delta_h.sum(axis=0)[:, None]

    def get_err(self, O):
        Y_pred = util.predict(O)
        return 1.0 - metrics.accuracy_score(self.labels, Y_pred)


    def predict(self, X):
        X, S_h, H, S_o, O = self.forward_pass()
        #y_pred = np.argmax(S, axis=1)



