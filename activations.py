import numpy as np
import pdb

# ----------------------- SOFTMAX ---------------------
def softmax(z, verbose=False):
    """ Exp-normalize trick from timvieira blog. """
    e_x             = np.exp(z - np.amax(z, axis=1)[:, None])
    out_layer_sum   = e_x.sum(axis=1)
    if verbose:
        print("\t\t softmax exp shape", e_x.shape)
        print("\t\t softmax sum shape", out_layer_sum.shape)
    assert(e_x.shape == z.shape)
    assert(out_layer_sum.shape[0] == z.shape[0])
    return e_x / out_layer_sum[:, None]

def _softmax(z_j, norm):
    """
    Args:
        z -- np.array of (weighted-sum) inputs to the output units.
    """
    return np.exp(z_j) / norm

def softmax_deriv(j, i, z):
    """
    Returns:
        derivative of g_j(z) w.r.t. z_i, where . . .
    Args:
        z -- np.array of (weighted-sum) inputs to the output units.
    """
    norm = sum(np.exp(z))
    g_j = _softmax(z[j], norm)
    if j == i:
        return g_j * (1 - g_j)
    else:
        return - g_j * _softmax(z[i], norm)

# ----------------------- RELU ---------------------

def relu(x):
    """
    Args:
        x -- array-like
    """
    return np.where(x > 0, x, 0)

def relu_deriv(x):
    return np.where(x > 0, 1, 0)

# ----------------------- SIGMOID ---------------------

def sigmoid(x):
    if np.all(x > 0):
        z = np.exp(-x)
        return 1. / (1. + z)
    else:
        z = np.exp(x)
        return z / (1. + z)

def sigmoid_deriv(x):
    return sigmoid(x) * (1. - sigmoid(x))


