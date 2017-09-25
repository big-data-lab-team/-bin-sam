from pyspark import SparkContext
from pyspark import SparkConf
conf = SparkConf().setAppName("tryrdd").setMaster("local[4]")
sc = SparkContext(conf=conf)

# lines = sc.textFile("a.txt")


#
# lineLengths = lines.map(lambda s:"({}------>{})".format(s, len(s)))
# totalLength = lineLengths.reduce(lambda a, b: a + b)
# print lineLengths.collect()

#
# pairs = lines.map(lambda s: (s, 1))
# counts = pairs.sortByKey()
#
# print counts.collect()


print sc.parallelize([[1, 2, 3, 4], [6,7,8,9], [10,11,12,13]]).reduce(lambda a,b:a+b)
