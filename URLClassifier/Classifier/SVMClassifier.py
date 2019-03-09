from pyspark.ml.classification import LinearSVC
from pyspark.sql import SparkSession as sess


class SVMClassifier:

    def __init__(self):
        pass

    def classify(self, max_passes, input_data):
        # Starting Spark.
        spark = sess.builder.appName("classify").getOrCreate();
        training_data = spark.read.text("trainingdata");

        lsvc = LinearSVC(maxIter=max_passes, regParam=0.1)

        return lsvc.fit(training_data)
