# util.py -- Compilation of helper functions we've used in past homeworks.
from scipy.ndimage import interpolation
import matplotlib.pyplot as plt
import numpy as np
import pdb

def one_hot(labels, num_classes=10):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    # Constructs a matrix where instead of number between 0...9 in each row,
    # has row vector of 0's except in the ith position, given that label = i.
    return np.eye(num_classes)[labels.astype(int)]

def predict(softmax_out):
    ''' From model and data points, output prediction vectors '''
    return np.argmax(softmax_out, axis=1)

def preprocess(X):
    #return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    #X = X/255.0
    return (X - np.mean(X, axis=0))

def log_transform(X):
    return np.log(X + 0.1)

def binarize(X):
    return np.where(X > 0, 1, 0)

def cross_entropy(O, Y_hot):
    """
    Args:
        O       -- N by n_out matrix of output unit values for N training examples.
        Y_hot   -- The corresponding one-hot encoded ground truth labels y_k.
    """
    n = O.shape[0]
    return - (1/n) *  np.sum(Y_hot * np.log(np.where(O > 0, O, 1e-30)), axis=1).sum()

def withBias(X):
    """
    Appends a column of 1s to X:
        (X.shape[0], X.shape[1]) ==> (X.shape[0], 1 + X.shape[1])
    """
    #pdb.set_trace()
    return np.hstack((np.ones((X.shape[0], 1)), X))


def weight_init(shape, how):
    if how == "random":
        return np.random.randn(*shape)
    elif how == "uniform":
        return np.random.uniform(-0.7, 0.7, shape) # ESL suggestion


def moments(image):
    # Credit: Dibya Ghosh
    c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
    totalImage = np.sum(image) #sum of pixels
    m0 = np.sum(c0*image)/totalImage #mu_x
    m1 = np.sum(c1*image)/totalImage #mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
    mu_vector = np.array([m0,m1]) # Notice that these are \mu_x,\mu_y respectively
    covariance_matrix = np.array([[m00,m01],[m01,m11]])
    return mu_vector, covariance_matrix

def deskew(image):
    # Credit: Dibya Ghosh
    c,v = moments(image)
    alpha = v[0,1]/v[0,0]
    affine = np.array([[1,0],[alpha,1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine,ocenter)
    return interpolation.affine_transform(image,affine,offset=offset)

def deskewAll(X):
    # Credit: Dibya Ghosh
    currents = []
    for i in range(len(X)):
        currents.append(deskew(X[i].reshape(28,28)).flatten())
    return np.array(currents)


def plot_error(iter_axis, loss, accuracy, params):

    plt.style.use('ggplot')
    fig = plt.figure()
    fig.suptitle("Gradient Descent", fontsize='x-large')

    # Left: Plot loss vs. iterations.
    ax1 = fig.add_subplot(121)
    ax1.plot(iter_axis, loss)
    ax1.set_xlabel('Number of Iterations')
    ax1.set_ylabel('Training Loss')

    bbox_props = dict(boxstyle="round", fc="w", ec="0.1", alpha=0.8)
    leg = r'$\eta$: {:.2E}'.format(params['eta']) + '\n' + \
    r'$\gamma$: {:2F}'.format(params['decay_const']) + '\n' + \
    r'n hid: {}'.format(params['n_hid']) +  '\n' + \
    r'$\lambda_2$: {:.2F}'.format(params['l2']) + '\n' + \
    r'batch size: {}'.format(params['batch_size']) + '\n' + \
    r'n epochs: {}'.format(params['n_epochs'])

    ax1.text(0.8 * iter_axis[-1], 0.9 * ax1.get_ylim()[1], leg,
            ha="center", va="center", size=12, bbox=bbox_props)

    #plt.text(0.8 * iter_axis[-1], 0.5 * np.max(loss), leg, fontsize='x-large')

    # Right: Plot accuracy vs. iterations.
    ax2 = fig.add_subplot(122)
    ax2.plot(iter_axis, accuracy)
    ax2.set_xlabel('Number of Iterations')
    ax2.set_ylabel('Classification Accuracy')

    title="LossAndAcc1"
    fig.savefig("png_dir/{0}.png".format(title))
    fig.savefig("pdf_dir/{0}.pdf".format(title))
    plt.show()

