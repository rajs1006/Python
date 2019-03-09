import re

import numpy as np
from pyspark.mllib.linalg import Vectors
from pyspark.sql import SparkSession as sess


class Data:

    def __init__(self, avg_word_weight):
        self.avg_word_weight = avg_word_weight;

    def __create_matrix__(self, data):
        word_avg = np.sum(
            [np.sum([k for l in re.split("https:|/", data) if g == l]) for g, k in
             self.avg_word_weight])
        word_count = sum([len(l) for l in re.split("https:|/", data)])
        return Vectors.dense([word_avg, word_count])

    def train_data(self, input_file):
        # Starting Spark.
        spark = sess.builder.appName("TrainData").getOrCreate();
        # '../urls'
        input_list = spark.read.text(input_file).rdd.map(lambda r: r[0]);
        # Training data
        training_data = input_list.map(lambda i: Data.__create_matrix__(self, i))

        training_data.saveAsTextFile("trainingdata")
        training_data = training_data.collect();
        # Spark stop
        spark.stop()

        return training_data

    def test_data(self, data):
        return Data.__create_matrix__(self, data)
