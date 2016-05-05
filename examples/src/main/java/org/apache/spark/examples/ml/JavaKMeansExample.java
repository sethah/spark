/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// scalastyle:off println
package org.apache.spark.examples.ml;

// $example on$
import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.VectorUDT;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
// $example off$

/**
 * An example demonstrating KMeans
 * Run with
 * <pre>
 * bin/run-example ml.JavaKMeansExample
 * </pre>
 */
public class JavaKMeansExample {

  public static void main(String[] args) {
    SparkSession spark = SparkSession
            .builder().appName("JavaKMeansExample").getOrCreate();

    // $example on$
    // Loads data
    List<Row> data = Arrays.asList(
      RowFactory.create(1, Vectors.dense(0.0, 0.0, 0.0)),
      RowFactory.create(2, Vectors.dense(0.1, 0.1, 0.1)),
      RowFactory.create(3, Vectors.dense(0.2, 0.2, 0.2)),
      RowFactory.create(4, Vectors.dense(9.0, 9.0, 9.0)),
      RowFactory.create(5, Vectors.dense(9.1, 9.1, 9.1)),
      RowFactory.create(6, Vectors.dense(9.2, 9.2, 9.2))
    );
    StructType schema = new StructType(new StructField[]{
      new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
      new StructField("features", new VectorUDT(), false, Metadata.empty())
    });
    Dataset<Row> dataset = spark.createDataFrame(data, schema);

    // Trains a k-means model
    KMeans kmeans = new KMeans()
      .setK(2)
      .setSeed(42L);
    KMeansModel model = kmeans.fit(dataset);

    // Shows the result
    System.out.println("Within Set Sum of Squared Errors = " + model.computeCost(dataset));
    Vector[] centers = model.clusterCenters();
    System.out.println("Cluster Centers: ");
    for (Vector center: centers) {
      System.out.println(center);
    }
    // $example off$

    spark.stop();
  }
}
