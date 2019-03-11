import re

from pyspark.sql import SparkSession as sess

from utils import SVMConstants


class WordAverage:

    def __init__(self, input_file):
        self.input_file = input_file

    def word_avg(self):
        # Starting Spark.
        spark = sess.builder.appName("wordcount").getOrCreate();
        # prepare input list from data file
        input_list = spark.read.text(self.input_file).rdd \
            .map(lambda r: re.split(SVMConstants.DATA_LABEL_SPLITTER, r[0])[0])
        # Process words from URL
        words = input_list \
            .flatMap(lambda url: re.split(SVMConstants.URL_SPLITTER, url)) \
            .filter(lambda w: len(w) > 0)

        total_word_count = words.count()
        avg_word_weight = words \
            .map(lambda w: (w, round(1 / total_word_count, 3))) \
            .reduceByKey(lambda accum, b: accum + b) \
            .collect()

        spark.stop()

        # Return the weight.
        return avg_word_weight
