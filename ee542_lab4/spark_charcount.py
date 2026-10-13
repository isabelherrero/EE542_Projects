from pyspark import SparkContext

sc = SparkContext(appName="CharCount")

text = sc.textFile("hdfs://master:9000/gutenberg/*.txt")

chars = text.flatMap(lambda line: list(line)) \
            .map(lambda ch: (ch, 1)) \
            .reduceByKey(lambda a, b: a + b)

chars.saveAsTextFile("hdfs://master:9000/output_spark_charcount")

sc.stop()
