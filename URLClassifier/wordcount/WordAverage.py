from pyspark.sql import SparkSession as sess
import re

class WordAverage:

    def __init__(self, input_file):
        self.input_file = input_file;

    def get_input(self):
        return self.input_list;

    def word_avg(self):
        # Starting Spark.
        spark = sess.builder.appName("wordcount").getOrCreate();
        # '../urls'
        input_list = spark.read.text(self.input_file).rdd.map(lambda r: r[0]);

        words = input_list \
            .flatMap(lambda url: re.split("https:|/", url)) \
            .filter(lambda w: len(w) > 0)

        total_word_count = words.count()
        avg_word = words \
            .map(lambda w: (w, round(1 / total_word_count, 3))) \
            .reduceByKey(lambda accum, b: accum + b)

        avg_word_weight = avg_word.collect()
        spark.stop();

        # Return the weight.
        return avg_word_weight;
