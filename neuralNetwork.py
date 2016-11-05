# Let NeuralNetwork deal with details like bias.
import sklearn.metrics as metrics
import numpy as np
import pdb
from networkComponents import *
import activations
import util

class NeuralNetwork(object):


    def __init__(self, n_in=784, n_hid=200, n_out=10,
                       eta=1e-4, decay_const=0.5, alpha=0.9, l2=0.01, batch_size=1, n_epochs=5):
        """
        Instance Variables:
            self.n_in   number of inputs (features) to network.
            self.n_hid  number of hidden units in single hidden layer.
            self.n_out  number of output neurons.
        """
        self.n_in       = n_in
        self.n_hid      = n_hid
        self.n_out      = n_out
        self.eta        = eta
        self.alpha      = alpha # momentum
        self.l2         = l2
        self.decay_const= decay_const
        self.batch_size = batch_size
        self.n_epochs   = n_epochs

        # Weight initialization.
        self.V = util.weight_init((self.n_hid, self.n_in + 1),  how="uniform")
        self.W = util.weight_init((self.n_out, self.n_hid + 1), how="uniform")
        self.prev_dV = np.zeros(self.V.shape)
        self.prev_dW = np.zeros(self.W.shape)

    def set_data(self, X_train, labels_train):
        """ Store data in instance attribute self.data,
            which is a dictionary with keys 'X' and 'Y'. """

        self.data                = self._get_shuffled_data_dict(X_train, labels_train)
        n_train = int(50e3)
        self.data['X_val']       = util.preprocess(self.data['X_train'][n_train:])
        self.data['X_train']     = util.preprocess(self.data['X_train'][:n_train])
        self.data['labels_val']  = self.data['labels_train'][n_train:]
        self.data['labels_train']= self.data['labels_train'][:n_train]

        self.X_active       = self.data['X_train']
        self.n_data_active  = self.X_active.shape[0]
        self.labels_active  = self.data['labels_train']

    def _get_shuffled_data_dict(self, X_train, labels_train):
        # First zip the data along the same axis before shuffling.
        joined_data = np.concatenate((X_train, labels_train[:, None]), axis=1)
        # Now we can shuffle the data together.
        np.random.shuffle(joined_data)
        # Separate back and return in a dictionary.
        (new_x, new_y) = joined_data[:, :self.n_in], joined_data[:, self.n_in:]
        return {'X_train': new_x, 'labels_train': new_y.flatten()}

    def set_active(self, indices=None):
        if indices is None: return
        self.X_active      = self.data['X_train'][indices]
        self.labels_active = self.data['labels_train'][indices]
        self.n_data_active = self.X_active.shape[0]


    def new_epoch(self, i):
        #self.eta /= (1 + 0.00001*i)
        self.eta *= self.decay_const


    def forward_pass(self, verbose=False, debug=False):
        """
        Computes the values of all units (neurons) over ALL sample points (vectorized).

        Args:
        Returns:
        """
        if (verbose): print("\n _______ Forward Pass _______")

        # Zeroth step: Outputs of input units.
        X = self.X_active

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

        self.O = O
        return X, S_h, H, S_o, O


    def get_deltas(self, O, H):
        """
        Returns:
            delta_o, delta_1,
        """
        from activations import relu_deriv
        delta_o = O - util.one_hot(self.labels_active)
        delta_h = relu_deriv(H) * (delta_o @ self.W[:, 1:])
        return delta_o, delta_h

    def train(self, X, H, O):
        delta_o, delta_h = self.get_deltas(O, H)

        #pdb.set_trace()
        norm = self.eta / self.n_data_active
        dW = np.zeros(self.W.shape)
        dW[:, 1:] = - norm * (delta_o.T @ H + self.l2 * self.W[:, 1:])
        dW[:, :1] = - norm * delta_o.sum(axis=0)[:, None]
        self.W += dW + self.alpha * self.prev_dW

        dV = np.zeros(self.V.shape)
        dV[:, 1:] = - norm * (delta_h.T @ X + self.l2 * self.V[:, 1:])
        dV[:, :1] = - norm * delta_h.sum(axis=0)[:, None]
        self.V += dV + self.alpha * self.prev_dV

        self.prev_dW = dW
        self.prev_dV = dV


    # =================== Evaluation =================

    def get_loss(self):
        return util.cross_entropy(self.O, util.one_hot(self.labels_active))

    def train_accuracy(self, total=False):
        if total:
            self.X_active = self.data['X_train']
            self.labels_active = self.data['labels_train']
            self.forward_pass()
        return metrics.accuracy_score(self.labels_active, util.predict(self.O))

    def val_accuracy(self):
        self.X_active       = self.data['X_val']
        self.labels_active  = self.data['labels_val']
        self.forward_pass()
        return metrics.accuracy_score(self.labels_active, util.predict(self.O))

    def predict_test(self, X_test):
        """ Return predictions given some unseen input data X_test."""
        self.X_active = X_test
        self.forward_pass()
        return util.predict(self.O)

    def print_results(self):
        f = open('results.txt', 'a')
        f.write("\n\n________________ NeuralNetwork::print_results() ________________")
        f.write("\nArchitecture:")
        f.write("\n\tHidden Units: {}".format(self.n_hid))
        f.write("\nHyperparameters:")
        f.write("\n\tLearning rate: {:.5E}".format(self.eta))
        f.write("\n\tdecay_const: {:.3F}".format(self.decay_const))
        f.write("\n\tMomentum: {:.2F}".format(self.alpha))
        f.write("\n\tl2 reg: {:.3F}".format(self.l2))
        f.write("\n\tBatch size:{}".format(self.n_data_active))
        f.write("\n\tNum Epochs:{}".format(self.n_epochs))
        f.write("\nEvaluations:")
        f.write("\n\tTraining Accuracy:{:.4f}".format(self.train_accuracy(total=True)))
        f.write("\n\tTest Accuracy:{:.4f}".format(self.val_accuracy()))
        f.close()

    def get_params(self):
        return {'eta':          self.eta,
                'decay_const':  self.decay_const,
                'n_hid':        self.n_hid,
                'l2':           self.l2,
                'batch_size':   self.batch_size,
                'n_epochs':     self.n_epochs}





