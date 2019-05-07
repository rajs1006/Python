import numpy as np


class Normalize:

    @staticmethod
    def normalize(Y, R):
        m, n = Y.shape
        YMean = np.zeros((m, 1))
        YNorm = np.zeros(Y.shape)

        for i in range(0, m):
            idx = np.where(R[i, :] == 1)
            YMean[i] = np.mean(Y[i, idx])
            YNorm[i, idx] = Y[i, idx] - YMean[i]

        return YNorm, YMean
