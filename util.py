# util.py -- Compilation of helper functions we've used in past homeworks.
import numpy as np

def one_hot(labels, num_classes=10):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    # Constructs a matrix where instead of number between 0...9 in each row,
    # has row vector of 0's except in the ith position, given that label = i.
    return np.eye(num_classes)[labels]

def predict(softmax_out):
    ''' From model and data points, output prediction vectors '''
    return np.argmax(softmax_out, axis=1)

def standardized(X):
    #return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return (X - np.mean(X, axis=0))

def log_transform(X):
    return np.log(X + 0.1)

def binarize(X):
    return np.where(X > 0, 1, 0)

def cross_entropy(O, labels):
    """
    Args:
        outputs -- The 'z_k(x)' output units for a given training example.
        labels  -- The corresponding one-hot encoded ground truth labels y_k.
    """
    print("O.shape", O.shape)
    print("labels.shape", labels.shape)
    #predictions = predict(O)
    O = predict(O)
    print("O.shape", O.shape)
    return - labels * np.log(np.where(O > 0, O, 1e-30))

def withBias(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))

#def withBias(X):
