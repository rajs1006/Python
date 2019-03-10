# using agg as plotter
import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt
import numpy as np

from classifier.SVMClassifier import SVMClassifier as SVM
from utils import SVMConstants


class PlotData:

    @staticmethod
    def plot_data(self, X, y):
        pos = np.where(y == 1);
        neg = np.where(y == 0);

        plt.plot(X[pos, 0], X[pos, 1], marker='h', color='r')
        plt.plot(X[neg, 0], X[neg, 1], marker='o', color='b')
        plt.savefig(SVMConstants.DATA_PICTURE_FILE)

    @staticmethod
    def plot_boundary(self, X, y, model, resolution=100):
        x1plot = np.linspace(min(X[:, 1]), max(X[:, 1]), resolution)
        x2plot = np.linspace(min(X[:, 2]), max(X[:, 2]), resolution)

        [X1, X2] = np.meshgrid(x1plot, x2plot)

        vals = np.zeros(X1.shape);
        for i in range(np.size(X1, 1)):
            this_X = [X1[:, i], X2[:, i]]
            vals[:, i] = SVM.classify(this_X, model)

        plt.contour(X1, X2, vals, (-1, 0, 1), 'b')
