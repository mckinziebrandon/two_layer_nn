import numpy as np

def sigmoid(x):
    if np.all(x > 0):
        z = np.exp(-x)
        return 1. / (1. + z)
    else:
        z = np.exp(x)
        return z / (1. + z)

def sigmoid_deriv(x):
    return sigmoid(x) * (1. - sigmoid(x))

def softmax(j, z):
    """
    Args:
        j -- subscript denoting this is for the jth output unit.
        z -- np.array of (weighted-sum) inputs to the output units.
    """
    return np.exp(z[j]) / sum(np.exp(z))

def softmax_deriv(j, i, z):
    """
    Returns:
        derivative of g_j(z) w.r.t. z_i, where . . .
    Args:
        j -- index applied to numerator:    g_j(z)
        i -- index applied to denominator:  z_i
        z -- np.array of (weighted-sum) inputs to the output units.
    """
    g_j = softmax(j, z)
    if j == i:
        return g_j * (1 - g_j)
    else:
        return - g_j * softmax(i, z)

