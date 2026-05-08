# Databricks notebook source
# MAGIC %md
# MAGIC # Task 5: User Segmentation and RMSE Evaluation using Spark
# MAGIC
# MAGIC This notebook represents Student 5 work in the recommendation system project.
# MAGIC
# MAGIC The task will be implemented using Apache Spark and Spark MLlib inside Databricks.
# MAGIC
# MAGIC Main responsibilities:
# MAGIC - Apply K-Means Clustering to group similar users.
# MAGIC - Calculate RMSE to evaluate recommendation model accuracy.
# MAGIC
# MAGIC Current step:
# MAGIC - Load and verify the MovieLens 25M dataset using Spark.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Spark Imports
# MAGIC
# MAGIC This section imports only Spark-related tools.
# MAGIC
# MAGIC No Pandas or scikit-learn will be used.

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import avg, count, col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Load MovieLens 25M Dataset
# MAGIC
# MAGIC In this step, we load the ratings and movies datasets from Databricks storage.
# MAGIC
# MAGIC The data is stored in Databricks Volumes


# COMMAND ----------


ratings_path = "/Volumes/workspace/default/ml25m/ratings.csv"
movies_path = "/Volumes/workspace/default/ml25m/movies.csv"

ratings_df = spark.read.csv(
    ratings_path,
    header=True,
    inferSchema=True
)
movies_df = spark.read.csv(
    movies_path,
    header=True,
    inferSchema=True
)


# COMMAND ----------

ratings_df.printSchema()
movies_df.printSchema()

# COMMAND ----------

ratings_count = ratings_df.count()
movies_count = movies_df.count()
print("Number of ratings: ",ratings_count)
print("Number of movies: ",movies_count)

# COMMAND ----------

display(ratings_df.limit(5))
display(movies_df.limit(5))

# COMMAND ----------


# MAGIC %md
# MAGIC ## Step 3: Clean Required Columns
# MAGIC
# MAGIC In this step, we keep only the columns needed for Task 5.
# MAGIC
# MAGIC Required columns:
# MAGIC - userId
# MAGIC - movieId
# MAGIC - rating
# MAGIC
# MAGIC We also remove null values to avoid errors in Spark MLlib.


# COMMAND ----------

ratings_clean_df = ratings_df.select(
    col("userId").cast("int"),
    col("movieId").cast("int"),
    col("rating").cast("double")
).dropna()
print("Original ratings count:", ratings_df.count())
print("Clean ratings count:", ratings_clean_df.count())

# COMMAND ----------

display(ratings_clean_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: User Feature Engineering
# MAGIC
# MAGIC In this step, we create behavioral features for each user.
# MAGIC
# MAGIC Features:
# MAGIC - avg_rating: the average rating given by the user
# MAGIC - rating_count: the total number of ratings made by the user


# COMMAND ----------

user_features_df = ratings_clean_df.groupBy("userId").agg(
    avg("rating").alias("avg_rating"),
    count("rating").alias("rating_count")
)

# COMMAND ----------

user_features_df.printSchema()
display(user_features_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Build Feature Vector
# MAGIC
# MAGIC In this step, we combine the user behavior columns into one feature vector.
# MAGIC
# MAGIC Spark MLlib requires a single vector column called features before applying machine learning algorithms

# COMMAND ----------

feature_assembler = VectorAssembler(
    inputCols=["avg_rating", "rating_count"],
    outputCol="features"
)

user_vector_df = feature_assembler.transform(user_features_df)

display(user_vector_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Feature Scaling
# MAGIC
# MAGIC In this step, we scale the feature vector.
# MAGIC
# MAGIC Scaling is important because rating_count can be much larger than avg_rating.
# MAGIC Without scaling, K-Means may focus mostly on rating_count and ignore avg_rating.


# COMMAND ----------


scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features",
    withMean=True,
    withStd=True
)
# mean and standard deviation are computed during training
scaler_model = scaler.fit(user_vector_df)

user_scaled_df = scaler_model.transform(user_vector_df)

display(user_scaled_df.limit(10))

# COMMAND ----------

# MAGIC ## Step 7: K-Means Clustering
# MAGIC
# MAGIC apply K-Means clustering to group users based on their behavior.
# MAGIC
# MAGIC The model uses scaled_features, which includes:
# MAGIC - avg_rating
# MAGIC - rating_count

# COMMAND ----------

kmeans = KMeans(
    featuresCol="scaled_features",
    predictionCol="cluster",
    k=3,
    seed=42
)

kmeans_model = kmeans.fit(user_scaled_df)
user_clusters_df = kmeans_model.transform(user_scaled_df)

display(user_clusters_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Cluster Summary
# MAGIC
# MAGIC we summarize each user cluster.
# MAGIC
# MAGIC The goal is to understand the behavior of users inside each cluster.

# COMMAND ----------

cluster_summary_df = user_clusters_df.groupBy("cluster").agg(
    count("userId").alias("users_count"),
    avg("avg_rating").alias("cluster_avg_rating"),
    avg("rating_count").alias("cluster_avg_rating_count")
).orderBy("cluster")

display(cluster_summary_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # MAGIC %md
# MAGIC # MAGIC ## Step 9: Cluster Interpretation
# MAGIC # MAGIC
# MAGIC # MAGIC Interpret clusters based on average rating and user activity
# MAGIC display(
# MAGIC     cluster_summary_df.orderBy("cluster_avg_rating_count", ascending=False)
# MAGIC )

# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster Results Interpretation
# MAGIC
# MAGIC The K-Means model grouped users into 3 clusters based on:
# MAGIC - Average rating behavior
# MAGIC - Number of ratings made by each user
# MAGIC
# MAGIC Cluster meanings:
# MAGIC - Cluster 0: Users with similar rating behavior and activity level.
# MAGIC - Cluster 1: Users with different rating behavior or activity level.
# MAGIC - Cluster 2: Users with another distinct behavior pattern.
# MAGIC
# MAGIC The exact meaning of each cluster depends on the values shown in the cluster summary table.

# COMMAND ----------

# MAGIC %md
# MAGIC # MAGIC %md
# MAGIC # MAGIC ## Step 10: RMSE Evaluation using ALS
# MAGIC # MAGIC
# MAGIC # MAGIC In this step, we train a temporary ALS model to evaluate recommendation accuracy.
# MAGIC # MAGIC
# MAGIC # MAGIC RMSE measures the difference between actual ratings and predicted ratings.
# MAGIC # MAGIC Lower RMSE means better prediction accuracy.

# COMMAND ----------

tran_df, test_df = ratings_clean_df.randomSplit([0.8, 0.2], seed=42)

als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    #Hidden Features
    rank=10,
    maxIter=10,
    #Reduces the problem of overfitting
    regParam=0.01,
    coldStartStrategy="drop",
    seed=42
)

als_model = als.fit(tran_df)

predictions_df = als_model.transform(test_df)

rmse = rmse_evaluator.evaluate(predictions)

display(predictions.limit(10))


# COMMAND ----------

# MAGIC %md
# MAGIC # MAGIC %md
# MAGIC # MAGIC ## Step 11: Calculate RMSE
# MAGIC # MAGIC
# MAGIC # MAGIC RMSE is used to evaluate the accuracy of the recommendation model.

# COMMAND ----------


rmse_evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

rmse = rmse_evaluator.evaluate(predictions_df)

print("Root Mean Squared Error (RMSE):", rmse)

# COMMAND ----------

# MAGIC %md
# MAGIC The ALS recommendation model achieved an RMSE of 0.8134. This means that, on average, the predicted ratings differ from the actual ratings by about 0.81 rating point. A lower RMSE indicates better recommendation accuracy, so this result shows that the model provides reasonable prediction performance.

# COMMAND ----------

# MAGIC %md
# MAGIC # COMMAND ----------
# MAGIC
# MAGIC # MAGIC %md
# MAGIC # MAGIC ## Final Task 5 Summary
# MAGIC # MAGIC
# MAGIC # MAGIC In this notebook, Task 5 was completed using Apache Spark and Spark MLlib.
# MAGIC # MAGIC
# MAGIC # MAGIC The main goals were:
# MAGIC # MAGIC - Apply K-Means Clustering to segment users based on behavior.
# MAGIC # MAGIC - Evaluate recommendation accuracy using RMSE.
# MAGIC # MAGIC
# MAGIC # MAGIC User features were created from the ratings dataset:
# MAGIC # MAGIC - avg_rating: average rating given by each user
# MAGIC # MAGIC - rating_count: number of ratings made by each user
# MAGIC # MAGIC
# MAGIC # MAGIC K-Means was applied with k = 3 after scaling the features.
# MAGIC # MAGIC The clusters showed different user behavior patterns based on rating activity and average rating tendency.
# MAGIC # MAGIC
# MAGIC # MAGIC A temporary ALS recommendation model was trained to calculate RMSE.
# MAGIC # MAGIC The model achieved:
# MAGIC # MAGIC
# MAGIC # MAGIC **RMSE = 0.8134**
# MAGIC # MAGIC
# MAGIC # MAGIC This means that the model predictions differ from the actual ratings by about 0.81 rating point on average.
# MAGIC # MAGIC Lower RMSE indicates better recommendation accuracy.

# COMMAND ----------

# MAGIC %md
# MAGIC # COMMAND ----------
# MAGIC
# MAGIC # MAGIC %md
# MAGIC # MAGIC ## Dependency Note
# MAGIC # MAGIC
# MAGIC # MAGIC The RMSE evaluation in this notebook was calculated using a temporary ALS baseline model.
# MAGIC # MAGIC
# MAGIC # MAGIC In the final integrated project, this evaluation can be replaced with the predictions generated by Student 4's ALS model.
# MAGIC # MAGIC
# MAGIC # MAGIC Required columns for final RMSE evaluation:
# MAGIC # MAGIC - userId
# MAGIC # MAGIC - movieId
# MAGIC # MAGIC - rating
# MAGIC # MAGIC - prediction