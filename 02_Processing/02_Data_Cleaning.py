# Databricks notebook source
ratings_path = "/Volumes/workspace/default/raw_data/ratings.dat"
movies_path = "/Volumes/workspace/default/raw_data/movies.dat"
users_path = "/Volumes/workspace/default/raw_data/users.dat"

users_df = (
    spark.read
    .option("sep", "::")
    .option("inferSchema", "true")
    .csv(users_path)
    .toDF("userId", "gender", "age", "occupation", "zipCode")
)

ratings_df = (
    spark.read
    .option("sep", "::")
    .option("inferSchema", "true")
    .csv(ratings_path)
    .toDF("userId", "movieId", "rating", "timestamp")
)

movies_df = (
    spark.read
    .option("sep", "::")
    .option("inferSchema", "true")
    .csv(movies_path)
    .toDF("movieId", "title", "genres")
)

ratings_df.show(5)
movies_df.show(5)
users_df.show(5)

# COMMAND ----------

ratings_df.describe().show()
movies_df.describe().show()
users_df.describe().show()

# COMMAND ----------

# DBTITLE 1,Cell 4
from pyspark.sql.functions import count, when, col

print("Number of Rows:", ratings_df.count())
print("Number of Columns:", len(ratings_df.columns))

ratings_df.printSchema()

ratings_df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in ratings_df.columns
]).show()

print("Movies Rows:", movies_df.count())
print("Movies Columns:", len(movies_df.columns))

movies_df.printSchema()

movies_df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in movies_df.columns
]).show()

print("Users Rows:", users_df.count())
print("Users Columns:", len(users_df.columns))
users_df.printSchema()
users_df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in users_df.columns
]).show()

# COMMAND ----------

from pyspark.sql.functions import split, col, from_unixtime

# 1. Your original cleaning logic
ratings_clean = ratings_df.dropDuplicates().dropna()
users_clean = users_df.dropDuplicates().dropna()

# 2. Movie Improvement: Split genres so we can analyze them individually
movies_clean = movies_df.dropDuplicates().dropna().withColumn(
    "genres_list", split(col("genres"), "\|")
)

# 3. Ratings Improvement: Convert timestamp to a readable Date format
ratings_clean = ratings_clean.withColumn(
    "rating_date", from_unixtime(col("timestamp")).cast("date")
)

print("✅ Feature Engineering Complete: Genres split and dates formatted!")

# COMMAND ----------

# 1. Clear the old, basic tables to make room for the new ones
spark.sql("DROP TABLE IF EXISTS clean_ratings")
spark.sql("DROP TABLE IF EXISTS clean_movies")
spark.sql("DROP TABLE IF EXISTS clean_users")

# 2. Save your new, expert-level tables
ratings_clean.write.mode("overwrite").saveAsTable("clean_ratings")
movies_clean.write.mode("overwrite").saveAsTable("clean_movies")
users_clean.write.mode("overwrite").saveAsTable("clean_users")

print("✅ SUCCESS: Expert tables are now saved and ready!")

# COMMAND ----------

spark.table("clean_ratings").show(5)

spark.table("clean_movies").show(5)

spark.table("clean_users").show(5)

# COMMAND ----------

print("Unique Users:",
      ratings_clean.select("userId").distinct().count())

print("Unique Movies:",
      ratings_clean.select("movieId").distinct().count())

print("Total Ratings:",
      ratings_clean.count())