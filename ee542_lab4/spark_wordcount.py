from pyspark import SparkContext

sc = SparkContext(appName="WordCount")

text = sc.textFile("hdfs://master:9000/gutenberg/*.txt")

counts = text.flatMap(lambda line: line.split()) \
             .map(lambda word: (word, 1)) \
             .reduceByKey(lambda a, b: a + b)

counts.saveAsTextFile("hdfs://master:9000/output_spark_wordcount")

sc.stop()
