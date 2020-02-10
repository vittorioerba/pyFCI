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

def FCI(dataset):
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

def FCI_MC(dataset,samples=500):
    """
    Compute the full correlation integral of the dataset by a Monte-Carlo random sampling

    :param dataset: a list of N d-dimensional vectors, i.e. a list of shape (N,d)
    :param samples: an integer that determines the number of pairs of points checked in the computation of the full correlation integral
    :returns: a list of shape (N(N-1)/2,2)
    """
    samples = int(min( len(dataset)*( len(dataset)-1 )/2, samples ))
    rs = np.empty(0)
    n = len(dataset)
    for k in range(samples):
        i = np.random.randint(0,n)
        j = np.random.randint(0,n)
        r = np.linalg.norm(dataset[i]-dataset[j])
        rs = np.append(rs, r)

    return np.transpose( [ np.sort(rs) , np.linspace(0,1,num=int(samples)) ] )

################################################################################

def analytical_FCI(x,d,x0=1):
    """
    Compute the analytical average full correlation integral on a d-dimensional sphere

    :param x: a real number, or a numpy vector of real numbers
    :param d: a real number, the dimension 
    :param x0: a real number (should be close to 1). It's such that f(x0)=0.5
    :returns: a real number, or a numpy vector of real numbers
    """
    return  0.5 * ( 1 + (scy.special.gamma((1+d)/2)) / (np.sqrt(np.pi) * scy.special.gamma(d/2) ) * (-2+(x/x0)**2) * scy.special.hyp2f1( 0.5, 1-d/2, 3/2, 1/4 * (-2+(x/x0)**2)**2 ) )

################################################################################

def fit_FCI(rho, samples=500):
    """
    Given a list of real 2D points in the domain = [0,2]x[0,1], it tries to fit them to the analytical_FCI curve.
    If the fit fails, it outputs [0,0,0]

    :param rho: real vector of shape (N,2)
    :param samples: an integer, the number of points of rho to be used in the fitting procedure
    :returns: a real number, or a numpy vector of real numbers
    """

    samples = min( len(rho),samples )
    data = rho[np.random.choice(len(rho),samples)]
    try:
        fit = scyopt.curve_fit( analytical_FCI, data[:,0], data[:,1] )
        mse = np.sqrt(np.mean([ (pt[1] - analytical_FCI(pt[0],fit[0][0],fit[0][1]))**2 for pt in data ]))
        return [fit[0][0]+1,fit[0][1],mse]
    except:
        return [0,0,0]



################################################################################

# TODO: modify to have also a cutoff by distance, and not only by kNN 

def local_FCI(dataset, center, ks):
    """
    xxxxxx

    At the moment, uses FCI_MC and fit FCI with default params

    :param dataset: a list of N d-dimensional vectors, i.e. a list of shape (N,d)
    :param center: the index of a vector in dataset
    :param ks: list of increasing integers. Number of neighbours around dataset[center] to be checked
    :returns:
    """
    neighbours = dataset[np.argsort(np.linalg.norm( dataset - dataset[center], axis=1))[0:ks[-1]]]    
  
    local = np.empty(shape=(0,5))
    for k in ks:
        fit = fit_FCI( FCI_MC( center_and_normalize( neighbours[0:k] ) ) )
        local = np.append(local, [[ k, np.linalg.norm( neighbours[k-1] - neighbours[0] ), fit[0], fit[1], fit[2] ]], axis=0 )

    return local


