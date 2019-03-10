from classifier.SVMClassifier import SVMClassifier
from data import PrepareData
from utils import SVMConstants
from wordcount import WordAverage


def main():
    test_input = 'https://www.netflix.com/watch/'
    file = SVMConstants.TRAINING_DATA_FILE

    w_w = WordAverage.WordAverage(file).word_avg()
    X, y = PrepareData.Data(w_w).train_data(file)
    x = PrepareData.Data(w_w).test_data(test_input)

    model = SVMClassifier().train(X, y)
    prediction = SVMClassifier().classify(x, model)

    print('\nProcessed %s\n\nURL Classified as: %s\n', test_input,
          'wrong url' if prediction == 0 else 'correct url');


main()
