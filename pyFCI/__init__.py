import numpy as np
import scipy as scy
import scipy.optimize as scyopt

################################################################################

def center_and_normalize(dataset):
    """
    Center and normalize a **dataset** of N d-dimensional points so that its Euclidean barycenter is the zero vector, and each of its points has norm 1. 

    :param dataset: vector of shape (N,d)
    :returns: vector of shape (N,d)
    """
    mean = np.mean(dataset,axis=0)
    return np.array([ vec / np.linalg.norm(vec) for vec in dataset - mean ])

################################################################################

def FCI(dataset):
    """
    Compute the full correlation integral of a **dataset** of N d-dimensional points by exact enumeration

    :param dataset: vector of shape (N,d)
    :returns: vector of shape (N(N-1)/2,2)
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
    Compute the full correlation integral of a **dataset** of N d-dimensional points by random sampling of **samples** pair of points

    :param dataset: vector of shape (N,d)
    :param samples: positive integer
    :returns: vector of shape (N(N-1)/2,2)
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
    Compute the analytical average full correlation integral on a **d**-dimensional sphere at **x**

    :param x: a real number in (0,2), or a vector of real numbers in (0,2)
    :param d: a real positive number
    :param x0: a real number (should be close to 1). It's such that f(x0)=0.5
    :returns: a real number, or a numpy vector of real numbers
    """
    return  0.5 * ( 1 + (scy.special.gamma((1+d)/2)) / (np.sqrt(np.pi) * scy.special.gamma(d/2) ) * (-2+(x/x0)**2) * scy.special.hyp2f1( 0.5, 1-d/2, 3/2, 1/4 * (-2+(x/x0)**2)**2 ) )

################################################################################

def fit_FCI(rho, samples=500):
    """
    Given an empirical full correlation integral **rho**, it tries to fit it to the analytical_FCI curve.
    To avoid slow-downs, only a random sample of **samples** points is used in the fitting.
    If the fit fails, it outputs [0,0,0]

    :param rho: vector of shape (N,2) of points in (0,2)x(0,1)
    :param samples: a positive integer
    :returns: the fitted dimension, the fitted x0 parameter and the mean square error between the fitted curve and the empirical points
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
    Given a **dataset** of N d-dimensional points, the index **center** of one of the points and a list of possible neighbourhoods **ks**, it estimates the local intrinsic dimension by using **fit_FCI()** of the reduced dataset of the first k-nearest-neighbours of **dataset[center]**, for each k in **ks**

    At the moment, it uses FCI_MC and fit FCI with default parameters

    :param dataset: a vector of shape (N,d)
    :param center: the index of a point in **dataset**
    :param ks: list of increasing positive integers
    :returns: a vector of shape (len(ks),5). For each k in **ks**, returns the list [ k, distance between dataset[center] and the k-th neighbour, fitted dimension, fitted x0, the mean square error of the fit ] 
    """
    neighbours = dataset[np.argsort(np.linalg.norm( dataset - dataset[center], axis=1))[0:ks[-1]]]    
  
    local = np.empty(shape=(0,5))
    for k in ks:
        fit = fit_FCI( FCI_MC( center_and_normalize( neighbours[0:k] ) ) )
        local = np.append(local, [[ k, np.linalg.norm( neighbours[k-1] - neighbours[0] ), fit[0], fit[1], fit[2] ]], axis=0 )

    return local


