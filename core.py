import numpy as np
import scipy as scy
import scipy.optimize as scyopt

################################################################################

def center_and_normalize(dataset):
    """
    Center and normalize a dataset so that its Euclidean barycenter is the zero vector, and each of its datapoints have norm 1. 

    :param dataset: a list of N d-dimensional vectors, i.e. a list of shape (N,d)
    :returns: a list of N d-dimensional, normalized vectors, with null barycenter
    """
    mean = np.mean(dataset,axis=0)
    return np.array([ vec / np.linalg.norm(vec) for vec in dataset - mean ])

################################################################################

def full_correlation_integral(dataset):
    """
    Compute the full correlation integral of the dataset by exact enumeration

    :param dataset: a list of N d-dimensional vectors, i.e. a list of shape (N,d)
    :returns: a list of shape (N(N-1)/2,2)
    """
    rs = np.empty(0)
    n = len(dataset)
    for i in range(n):
        for j in range(i+1,n):
            r = np.linalg.norm(dataset[i]-dataset[j])
            rs = np.append(rs, r)
    return np.transpose( [ np.sort(rs) , np.linspace(0,1,num=int(n*(n-1)/2)) ] )

################################################################################

def full_correlation_integral_MC(dataset,samples=500):
    """
    Compute the full correlation integral of the dataset by a Monte-Carlo random sampling

    :param dataset: a list of N d-dimensional vectors, i.e. a list of shape (N,d)
    :param samples: an integer that determines the number of pairs of points checked in the computation of the full correlation integral
    :returns: a list of shape (N(N-1)/2,2)
    """
    rs = np.empty(0)
    n = len(dataset)
    for k in range(samples):
        i = np.random.randint(0,n)
        j = np.random.randint(0,n)
        r = np.linalg.norm(dataset[i]-dataset[j])
        rs = np.append(rs, r)

    return np.transpose( [ np.sort(rs) , np.linspace(0,1,num=int(samples)) ] )

################################################################################

def analytical_full_correlation_integral(x,d,x0=1):
    """
    Compute the analytical average full correlation integral on a d-dimensional sphere

    :param x: a real number, or a numpy vector of real numbers
    :param d: a real number, the dimension 
    :param x0: a real number (should be close to 1). It's such that f(x0)=0.5
    :returns: a real number, or a numpy vector of real numbers
    """
    return  0.5 * ( 1 + (scy.special.gamma((1+d)/2)) / (np.sqrt(np.pi) * scy.special.gamma(d/2) ) * (-2+(x/x0)**2) * scy.special.hyp2f1( 0.5, 1-d/2, 3/2, 1/4 * (-2+(x/x0)**2)**2 ) )

################################################################################

def fit_full_correlation_integral(rho, samples=500):
    """
    xxxxxx

    :param rho: xxxxxxxx
    :param samples: an integer, the number of points of rho to be used in the fitting procedure
    :returns: a real number, or a numpy vector of real numbers
    """

    data = rho[np.random.choice(len(rho),samples)]
    return scyopt.curve_fit( analytical_full_correlation_integral, data[:,0], data[:,1] )

