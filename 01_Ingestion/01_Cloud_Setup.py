# Databricks notebook source
# Define all three paths from your Volume
ratings_path = "/Volumes/workspace/default/raw_data/ratings.dat"
movies_path = "/Volumes/workspace/default/raw_data/movies.dat"
users_path = "/Volumes/workspace/default/raw_data/users.dat"

# 1. Ingest Ratings
ratings_df = (spark.read
              .option("sep", "::")
              .option("inferSchema", "true")
              .csv(ratings_path)
              .toDF("userID", "movieID", "rating", "timestamp"))

# 2. Ingest Movies
movies_df = (spark.read
             .option("sep", "::")
             .option("inferSchema", "true")
             .csv(movies_path)
             .toDF("movieID", "title", "genres"))

# 3. Ingest Users (Added this to complete the set)
users_df = (spark.read
            .option("sep", "::")
            .option("inferSchema", "true")
            .csv(users_path)
            .toDF("userID", "gender", "age", "occupation", "zip-code"))

# Save all as RAW tables in the default database
ratings_df.write.mode("overwrite").saveAsTable("raw_ratings")
movies_df.write.mode("overwrite").saveAsTable("raw_movies")
users_df.write.mode("overwrite").saveAsTable("raw_users")

print("✅ INGESTION COMPLETE: 'raw_ratings', 'raw_movies', and 'raw_users' are ready for cleaning!")