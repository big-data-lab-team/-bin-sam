from pyspark import SparkContext
from pyspark import SparkConf

class CustomSparkContext(SparkContext):
    def customBinaryRecords(self, path, recordLength):
        br = sc.newAPIHadoopFile(
            path,
            'org.apache.spark.input.FixedLengthBinaryInputFormat',
            'org.apache.hadoop.io.LongWritable',
            'org.apache.hadoop.io.BytesWritable',
            conf={'org.apache.spark.input.FixedLengthBinaryInputFormat.recordLength':"{}".format(recordLength)})

        return br.filter(lambda (k,v):k>=1).flatMap(lambda (k,v): v).gro

Y = 3850
Z = 3025
X = 3500

y,z,x = (770, 605, 700)

conf = SparkConf().setAppName("tryspark").setMaster("local[4]")
sc = CustomSparkContext(conf=conf)
# data = '/data/bigbrain_1540_1815_0.nii'
data = '/data/bigbrain_1540_1815_0.nii'
# data  = "/data/bigbrain_40microns.nii"
rdd = sc.customBinaryRecords(data, 352)

# val data = br.map { case (k, v) =>
#       val bytes = v.getBytes
#       assert(bytes.length == recordLength, "Byte array does not have correct length")
#       bytes
#     }
print rdd.take(100)



# header = rdd.first()
#
# rows = rdd.filter(lambda h: h != header)
#
#
# rows.reduce(lambda a,b:a+b)
# def f(idx, iter):
#     if (idx == 0):
#         iter
#
# print rdd.mapPartitionsWithIndex(f).take(1)



