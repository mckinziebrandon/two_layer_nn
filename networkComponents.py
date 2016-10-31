# networkComponents.py
import numpy as np

#LAYER_TYPES = {"INPUT": , "HIDDEN", "OUTPUT": softmax}

class Layer(object):

    def __init__(self, layer_type):
        if layer_type not in LAYER_TYPES:
            raise RuntimeError

        self.layer_type = layer_type



class Model(object):
    """
    A parameter tensor for a neural network.
    (i.e. the learned connections between layers)
    """

    def __init__(self, shape):
        """
        Args:
            shape -- tuple of (n_in, n_out) for model tensor.
        """
        self.shape = shape
        # default choice
        self.params = np.zeros(shape)
        # TODO: Decide initialization.



