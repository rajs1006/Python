from data import PrepareData
from wordcount import WordAverage


class Data:

    @staticmethod
    def load_data(train_data_file, test_data):
        # Train data.
        word_avg = WordAverage.WordAverage(train_data_file).word_avg()
        X, y = PrepareData.Data(word_avg).train_data(train_data_file)
        # Test data.
        x = PrepareData.Data(word_avg).test_data(test_data)

        return X, y, x
