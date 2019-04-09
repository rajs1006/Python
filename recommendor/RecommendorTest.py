import numpy as np
import scipy.io as spi
from scipy.optimize import minimize

from cofi.ColaborativeFilterOct import cofi
from cofi.ColaborativeFiltering import CollaborativeFilter
from cofi.Normalize import Normalize

if __name__ == '__main__':
    # Load data to be tested
    data = spi.loadmat('data')
    Y, R = data['Y'], data['R']

    # Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
    data2 = spi.loadmat('testdata.mat')
    X, Theta = data2['X'], data2['Theta']

    YNorm, YMean = Normalize.normalize(Y)

    numMovie, numUser = 5, 4

    numFeature = 3

    X = X[0:numMovie, 0: numFeature]
    Theta = Theta[0:numUser, 0: numFeature]
    Y = Y[0:numMovie, 0: numUser]
    R = R[0:numMovie, 0: numUser]

    initialParam = np.concatenate((X[:], Theta[:])).ravel()

    lam = 0
    J, Theta = cofi.fit(initialParam, Y, R, numUser, numMovie, numFeature, lam)
    print("for lambda = 0 , cost should be around 22.22 , and calculated value is : %d ", J)

    lam = 1.5
    J, Theta = cofi.fit(initialParam, Y, R, numUser, numMovie, numFeature, lam)
    print("for lambda = 1.5 , cost should be around 31.34 , and calculated value is : %d ", J)

    # Conjugate gradient function.
    options = {'maxiter': 0, 'disp': True}

    lam = 0
    args = (Y, R, numUser, numMovie, numFeature, lam)

    J = minimize(CollaborativeFilter.cost, initialParam, args=args, jac=CollaborativeFilter.gradient, method='CG', options=options)
    print("for lambda = 0 , cost should be around 22.22 , and calculated value is : %d ", J.fun)

    lam = 1.5
    args = (Y, R, numUser, numMovie, numFeature, lam)

    J = minimize(CollaborativeFilter.cost, initialParam, args=args, jac=CollaborativeFilter.gradient, method='CG', options=options)
    print("for lambda = 0 , cost should be around 31.34 , and calculated value is : %d ", J.fun)
