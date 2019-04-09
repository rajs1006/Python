# using agg as plotter
import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt
import numpy as np

from classifier.SVMClassifier import SVMClassifier as SVM
from utils import SVMConstants


class PlotData:

    @staticmethod
    def plot_data(X, y):
        pos = np.where(y >= 0)
        neg = np.where(y < 0)

        plt.scatter(X[pos, 0], X[pos, 1], marker='h', color='r')
        plt.scatter(X[neg, 0], X[neg, 1], marker='o', color='b')
        plt.savefig(SVMConstants.DATA_PICTURE_FILE)

    @staticmethod
    def plot_boundary(X, y, x, model, resolution=100):
        # Plot data first.
        PlotData.plot_data(X, y)

        x1plot = np.linspace(min(X[:, 0]), max(X[:, 0]), resolution)
        x2plot = np.linspace(min(X[:, 1]), max(X[:, 1]), resolution)

        [x1, x2] = np.meshgrid(x1plot, x2plot)

        vals = np.zeros(x1.shape)
        for i in range(np.size(x1, 1)):
            x = [x1[:, i], x2[:, i]]
            vals[:, i] = SVM.classify(np.array(x).T, model)

        plt.contour(x1, x2, vals, color='r')
        plt.savefig(SVMConstants.DATA_PICTURE_FILE)
