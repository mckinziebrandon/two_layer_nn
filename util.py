# util.py -- Compilation of helper functions we've used in past homeworks.
import numpy as np

def one_hot(labels, num_classes=10):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    # Constructs a matrix where instead of number between 0...9 in each row,
    # has row vector of 0's except in the ith position, given that label = i.
    return np.eye(num_classes)[labels]

def predict(model, X):
    ''' From model and data points, output prediction vectors '''
    return np.argmax(np.dot(X, model), axis=1)

def standardized(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def log_transform(X):
    return np.log(X + 0.1)

def binarize(X):
    return np.where(X > 0, 1, 0)

def cross_entropy(outputs, labels):
    """
    Args:
        outputs -- The 'z_k(x)' output units for a given training example.
        labels  -- The corresponding one-hot encoded ground truth labels y_k.
    """
    return - np.sum(labels * np.log(outputs))

def appendBiasDim(x):
    return np.append(np.ones(x[:, None].shape), x[:, None], axis=1)
