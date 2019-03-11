import numpy as np

from classifier.SMOModel import SMOModel


class SVMClassifier:

    @staticmethod
    def train(X, y, max_passes=5, tol=1e-3, C=1, b=0, kernel='linear'):
        m, n = X.shape
        # hyper parameter alpha.
        alphas = np.zeros((m, 1))
        E = np.zeros((m, 1))
        # Kernel
        K = X.dot(X.T)

        passes = 0
        # until max pass
        while passes < max_passes:
            num_changed_alphas = 0

            for i in range(m):

                E[i] = b + sum(((alphas.T * y.T) * K[:, i].T).T) - y[i]

                if (y[i] * E[i] < -tol and alphas[i] < C) or (y[i] * E[i] > tol and alphas[i] > 0):

                    j = np.int(np.ceil(m * np.random.random())) - 1
                    while i == j:
                        j = np.int(np.ceil(m * np.random.random())) - 1

                    E[j] = b + sum(((alphas.T * y.T) * K[:, j].T).T) - y[j]

                    alpha_i_old = alphas[i]
                    alpha_j_old = alphas[j]

                    if y[i] == y[j]:
                        L = max(0, alphas[j] + alphas[i] - C)
                        H = min(C, alphas[j] + alphas[i])
                    else:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(C, C + alphas[j] - alphas[i])

                    if L == H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]

                    if eta >= 0:
                        continue

                    # Measuring new value of alpha j
                    alphas[j] = alphas[j] - (y[j] * (E[i] - E[j])) / eta

                    # Clip
                    alphas[j] = min(H, alphas[j])
                    alphas[j] = max(L, alphas[j])

                    if abs(alphas[j]) - alpha_j_old < tol:
                        alphas[j] = alpha_j_old
                        continue

                    alphas[i] = alphas[i] + y[i] * y[j] * (alpha_j_old - alphas[j])

                    b1 = b - E[i] - y[i] * (alphas[i] - alpha_i_old) * K[i, j] - y[j] * (alphas[j] - alpha_j_old) \
                         * K[i, j]
                    b2 = b - E[j] - y[i] * (alphas[i] - alpha_i_old) * K[i, j] - y[j] * (alphas[j] - alpha_j_old) \
                         * K[j, j]

                    if 0 < alphas[i] < C:
                        b = b1
                    elif 0 < alphas[j] < C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2

                    num_changed_alphas = num_changed_alphas + 1

            if num_changed_alphas == 0:
                passes = passes + 1
            else:
                passes = 0

        idx, idy = np.where(alphas > 0)

        return SMOModel(X[idx, :], y[idx], C, kernel, alphas[idx], b, (alphas.T * y.T).dot(X))

    @staticmethod
    def classify(x, model):
        # handling a single data (row vector)
        m = SVMClassifier.__index__(x)

        prediction = np.zeros(m)

        p = x.dot(model.w.T) + model.b

        idx_p = SVMClassifier.__prediction_index__(p)

        prediction[idx_p] = 1

        return prediction

    @staticmethod
    def __index__(x):
        try:
            m, n = x.shape
        except:
            m = 1
        return m

    @staticmethod
    def __prediction_index__(p):
        try:
            idx_p, idy_p = np.where(p >= 0.1) or np.where(p <= -0.1)
        except:
            idx_p = p >= 0.1 or p <= -0.1
        return idx_p
