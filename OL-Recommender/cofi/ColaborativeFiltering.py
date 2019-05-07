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

        X_grad = np.zeros(X.shape)
        Theta_grad = np.zeros(Theta.shape)

        for i in range(0, numMovies):
            idx = np.where(R[i, :] == 1)
            ThetaTemp = Theta[idx, :]
            YTemp = Y[i, idx]
            X_grad[i, :] = (ThetaTemp.dot(X[i, :]) - YTemp).dot(ThetaTemp) + lam * X[i, :]

        for i in range(0, numUser):
            idx = np.where(R[:, i] == 1)
            XTemp = X[idx, :]
            YTemp = Y[idx, i]
            Theta_grad[i, :] = (XTemp.dot(Theta[i, :]) - YTemp).dot(XTemp) + lam * Theta[i, :]

        params = np.concatenate((X_grad, Theta_grad)).ravel()

        return params
