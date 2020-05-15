from pyspark import SparkContext
from pyspark.sql import *
from graphframes import *
import time
import os
import sys


# spark-submit --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 task1.py
os.environ["PYSPARK_SUBMIT_ARGS"] = ( "--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 pyspark-shell")

st = time.time()

sc = SparkContext('local[*]', 'A4_T1')
sc.setLogLevel("ERROR")
spark = SparkSession.builder.appName('LabelPropagation').getOrCreate()


ip_data  = sc.textFile(sys.argv[1]).persist()
edgesRdd = ip_data.flatMap(lambda x: [(x.split(" ")[0],x.split(" ")[1]),(x.split(" ")[1],x.split(" ")[0])]).collect()
nodes = ip_data.flatMap(lambda x: (x.split(" ")[0],x.split(" ")[1])).distinct().collect()


nodesRdd = list()
for i in nodes:
    nodesRdd.append((i,))


vertices = spark.createDataFrame(nodesRdd, ["id"])  # .toDF("id", "node")
edges = spark.createDataFrame(edgesRdd, ["src", "dst"])  # .toDF("src", "dst")


gf = GraphFrame(vertices, edges)

result = gf.labelPropagation(maxIter=5)

communities = result.rdd.map(list).map(lambda x: (x[1], [x[0]])).reduceByKey(lambda x, y: x + y).map(lambda x: sorted(x[1])).collect()

communities.sort()
communities.sort(key=len)

file = open(sys.argv[2], 'w')

for i in communities:
    comm = ', '.join("'" + str(item) + "'" for item in i)
    file.write(comm + "\n")

print("time: ", time.time() - st)