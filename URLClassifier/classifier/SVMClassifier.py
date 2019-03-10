import numpy as np

from classifier.SMOModel import SMOModel


class SVMClassifier:

    @staticmethod
    def train(X, Y, max_passes=5, tol=1e-3, C=1, b=0, kernel='linear'):
        m, n = X.shape

        alphas = np.zeros((m, 1))
        E = np.zeros((m, 1))

        K = X.dot(X.T)

        passes = 0

        while passes < max_passes:
            num_changed_alphas = 0

            for i in range(m):
                E[i] = b + sum(((alphas.T * Y.T) * K[:, i].T).T) - Y[i]

                if (Y[i] * E[i] < -tol and alphas[i] < C) or (Y[i] * E[i] > tol and alphas[i] > 0):

                    j = np.int(np.ceil(m * np.random.random())) - 1
                    if i == j:
                        j = np.int(np.ceil(m * np.random.random())) - 1

                    E[j] = b + sum(((alphas.T * Y.T) * K[:, j].T).T) - Y[j]

                    alpha_i_old = alphas[i]
                    alpha_j_old = alphas[j]

                    if Y[i] == Y[j]:
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

                    if abs(alphas[j]) - alpha_j_old < tol:
                        alphas[j] = alpha_j_old
                        continue

                    alphas[i] = alphas[i] + Y[i] * Y[j] * (alpha_j_old - alphas[j])

                    b1 = b - E[i] - Y[i] * (alphas[i] - alpha_i_old) * K[i, j] - Y[j] * (alphas[j] - alpha_j_old) \
                         * K[i, j]
                    b2 = b - E[j] - Y[i] * (alphas[i] - alpha_i_old) * K[i, j] - Y[j] * (alphas[j] - alpha_j_old) \
                         * K(j, j)

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

        return SMOModel(X[idx, :], Y[idx], C, kernel, alphas[idx], b, np.mean(((alphas.T * Y.T).T * X), axis=0))

    @staticmethod
    def classify(x, model):
        prediction = np.zeros(x.shape)

        p = x.dot(model.w) + model.b

        # prediction[np.where(p >= 0)] = 1

        return 1 if p >= 0 else 0
