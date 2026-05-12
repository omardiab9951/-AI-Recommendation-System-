# Databricks notebook source
# Load the cleaned "Silver" data we created in the previous notebook
df_movies = spark.table("clean_movies")
df_ratings = spark.table("clean_ratings")

print("Silver Tables loaded successfully for analysis!")

# COMMAND ----------

from pyspark.sql.functions import count, avg, col, round

# Join movies and ratings to see titles
top_movies = (df_ratings.join(df_movies, "movieId")
              .groupBy("title")
              .agg(count("rating").alias("total_ratings"), 
                   round(avg("rating"), 2).alias("avg_rating"))
              .filter("total_ratings > 100")
              .orderBy(col("total_ratings").desc())
              .limit(10))

display(top_movies)

# COMMAND ----------

from pyspark.sql.functions import explode

# Use the 'genres_list' we created during cleaning to see which genres are most common
genre_counts = (df_movies.withColumn("genre", explode("genres_list"))
                .groupBy("genre")
                .count()
                .orderBy(col("count").desc()))

display(genre_counts) 
# Note: Click the '+' icon in the Databricks output to turn this into a Pie Chart!