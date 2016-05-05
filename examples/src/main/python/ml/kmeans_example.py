#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function

# $example on$
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import DoubleType, StructField, StructType
# $example off$
from pyspark.sql import SparkSession

if __name__ == "__main__":

    spark = SparkSession.builder.appName("PythonKMeansExample").getOrCreate()

    # $example on$
    # load the data
    vecAssembler = VectorAssembler()\
        .setInputCols(["x", "y", "z"])\
        .setOutputCol("features")
    schema = StructType([
        StructField("x", DoubleType()),
        StructField("y", DoubleType()),
        StructField("z", DoubleType())
    ])
    dataset = vecAssembler.transform(
        spark
            .read
            .format("csv")
            .option("sep", " ")
            .schema(schema)
            .load("data/mllib/kmeans_data.txt")
    )

    kmeans = KMeans(k=2, seed=1)
    model = kmeans.fit(dataset)
    centers = model.clusterCenters()

    print("Within Set Sum of Squared Errors = %s" % model.computeCost(dataset))

    print("Final cluster centers:")
    for center in centers:
        print(center)
    # $example off$

    spark.stop()
