import argparse

from classifier.SVMClassifier import SVMClassifier
from data.Data import Data
from data.PlotData import PlotData as Plot
from utils import SVMConstants


def main():
    # Fetch input parameters.
    args = return_args()

    test_input = args.test_data_url

    X, y, x = Data.load_data(args.training_data_file, test_input)
    # Plot data
    Plot.plot_data(X, y)
    # SVM model.
    model = SVMClassifier().train(X, y)
    # print training details
    print('\nmodel.X: {}\nmodel.y: {}\nmodel.alphas: {}\nmodel.w: {}\n'
          .format(model.X, model.y, model.alphas, model.w))

    # prediction.
    prediction = SVMClassifier().classify(x, model)
    # print test details
    print('\nx: {}\nprediction: {}\n'.format(x, prediction))

    # Plot boundary
    Plot.plot_boundary(X, y, x, model)

    print('\nProcessed... {}\n\nURL Classified as: {}\n'
          .format(test_input, 'wrong url' if prediction == 0 else 'correct url'))


def return_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_data_url')
    parser.add_argument('training_data_file', nargs='?', default=SVMConstants.TRAINING_DATA_FILE)

    return parser.parse_args()


main()
