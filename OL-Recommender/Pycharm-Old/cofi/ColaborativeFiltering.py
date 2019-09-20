import numpy as np


class CollaborativeFilter:

    @staticmethod
    def cost(params, *args):
        # Args
        Y, R, numUser, numMovies, numFeature, lam = args
        X = np.reshape(params[0:numMovies * numFeature], (numMovies, numFeature))
        Theta = np.reshape(params[numMovies * numFeature:], (numUser, numFeature))

        J = sum(sum(R * np.square((Theta.dot(X.T)).T - Y))) / 2 + (lam * sum(sum(np.square(Theta)))) / 2 + (
                lam * sum(sum(np.square(X)))) / 2

        return J

    @staticmethod
    def gradient(params, *args):
        # Args
        Y, R, numUser, numMovies, numFeature, lam = args
        X = np.reshape(params[0:numMovies * numFeature], (numMovies, numFeature))
        Theta = np.reshape(params[numMovies * numFeature:], (numUser, numFeature))

        grad = np.multiply((np.dot(X, Theta.T) - Y), R)
        # gradients
        X_grad = np.dot(grad, Theta) + lam * X
        Theta_grad = np.dot(grad.T, X) + lam * Theta

        params = np.concatenate((X_grad, Theta_grad)).ravel()

        return params

