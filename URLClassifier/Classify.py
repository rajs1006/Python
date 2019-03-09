from wordcount import WordAverage
from data import PrepareData
from Classifier.SVMClassifier import SVMClassifier

def main():

    file = "urls"

    w_w = WordAverage.WordAverage(file).word_avg()
    training_d = PrepareData.Data(w_w).train_data(file)
    test_d = PrepareData.Data(w_w).test_data("https://www.netflix.com/watch/")

    model  = SVMClassifier().classify(10, training_d)

    print((test_d - training_d))


main()