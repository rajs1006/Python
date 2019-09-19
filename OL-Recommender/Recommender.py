import numpy as np
import numpy.random as rand
import scipy.io as spi
from scipy.optimize import minimize

from Utils.FileReader import FileReader
from cofi.ColaborativeFiltering import CollaborativeFilter
from cofi.Normalize import Normalize


def train():
    global YMean, R, X, Theta

    YNorm, YMean = Normalize.normalize(Y, R)
    numMovie, numUser = Y.shape
    numFeature = 10

    X = rand.rand(numMovie, numFeature)
    Theta = rand.rand(numUser, numFeature)
    initialParam = np.concatenate((X, Theta)).ravel()
    lam = 10

    options = {'maxiter': 1000, 'disp': True}
    args = (YNorm, R, numUser, numMovie, numFeature, lam)
    optimum = minimize(CollaborativeFilter.cost, initialParam, args=args, jac=CollaborativeFilter.gradient, method='CG',
                       options=options)

    print('Optimal cost is {} with status {} after {} iterations '.format(optimum.fun, optimum.success, optimum.nit))
    theta = optimum.x
    X = np.reshape(theta[0:numMovie * numFeature], (numMovie, numFeature))
    Theta = np.reshape(theta[numMovie * numFeature:], (numUser, numFeature))
    print(X.shape, Theta)

if __name__ == '__main__':
    # Load data to to train
    data = spi.loadmat('data')
    Y, R = data['Y'], data['R']
    print(Y.shape, R.shape)
    movieList = FileReader('movie_ids.txt').read_file()
    myMovieRating = np.zeros((len(movieList), 1))

    myMovieRating[0] = 4
    myMovieRating[97] = 2
    myMovieRating[6] = 3
    myMovieRating[11] = 5
    myMovieRating[53] = 4
    myMovieRating[63] = 5
    myMovieRating[65] = 3
    myMovieRating[68] = 5
    myMovieRating[182] = 4
    myMovieRating[225] = 5
    myMovieRating[354] = 5

    for i in range(0, len(myMovieRating)):
        if myMovieRating[i] != 0:
            print('Rated {} for movie {}'.format(int(myMovieRating[i]), movieList[i]))

    Y = np.concatenate((myMovieRating, Y), axis=1)
    R = np.concatenate(((myMovieRating > 0).astype(int), R), axis=1)

    # Train the optimizer
    train()
    np.mean(X, axis=0)
    p = X.dot(Theta.T)
    #p = np.reshape(np.mean(Theta, axis=0), (10, 1)).dot(np.reshape(np.mean(Theta, axis=0), (1, 10)))
    myPredictions = np.reshape(p[:, 0], (len(p), 1)) + YMean

    sortedIndex = np.argsort(myPredictions, axis=0)[::-1]

    print('\nTop Recommendations:- ')
    for i in range(0, 10):
        j = sortedIndex[i].item()
        print(' Rating {} for movie {}'.format(myPredictions[j], movieList[j]))

    trainedData = {'X': X, 'Theta': Theta}
    spi.savemat('traineddata', trainedData)
