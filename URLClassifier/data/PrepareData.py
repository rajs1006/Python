import re

import numpy as np
from pyspark.mllib.linalg import Vectors
from pyspark.sql import SparkSession as Session

from data.PlotData import PlotData
from utils import SVMConstants


class Data:

    def __init__(self, avg_word_weight):
        self.avg_word_weight = avg_word_weight

    @staticmethod
    def __splitter__(self, url):
        s = re.split(SVMConstants.DATA_LABEL_SPLITTER, url)
        return Data.__create_matrix__(self, s[0]), np.int(s[1])

    @staticmethod
    def __create_matrix__(self, data):
        word_avg = np.sum(
            [np.sum([k for l in re.split(SVMConstants.URL_SPLITTER, data) if g == l]) for g, k in
             self.avg_word_weight])
        word_count = sum([len(l) for l in re.split(SVMConstants.URL_SPLITTER, data)])
        return Vectors.dense([word_avg, word_count])

    def train_data(self, input_file):
        # Starting Spark.
        spark = Session.builder.appName("TrainData").getOrCreate()
        # '../trainingUrls'
        input_list = spark.read.text(input_file).rdd.map(lambda r: r[0])
        # Training data
        training_data = input_list \
            .map(lambda i: Data.__splitter__(self, i)) \
            .collect()

        X, y = zip(*training_data)
        X, y = np.array(X), np.array(y)

        # Spark stop
        spark.stop()
        # Plot data
        PlotData().plot_data(self, X, y)
        return X, y

    def test_data(self, data):
        return np.array(Data.__create_matrix__(self, data))
