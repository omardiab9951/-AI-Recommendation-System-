# Databricks notebook source
from pyspark.sql.functions import col, pow, mean, sqrt

# Load production recommendations
gold_recs = spark.table("workspace.default.gold_recommendations")

# Load movie titles from clean data
movie_titles = spark.table("clean_movies").select(
    col("movieId"),
    col("title"),
    col("genres")
)

print("Production data loaded successfully.")

# COMMAND ----------

def show_recommendations(target_user):
    user_data = gold_recs.filter(col("userId") == target_user)

    result = user_data.join(
        movie_titles,
        user_data.movie_1.cast("int") == movie_titles.movieId.cast("int"),
        "left"
    ).select(
        col("userId"),
        col("movie_1").alias("recommended_movieId"),
        col("title").alias("top_recommendation"),
        col("genres"),
        col("score_1").alias("match_score")
    )

    return result

# COMMAND ----------

selected_user = 1

print("===========================================")
print("Production Recommendations Dashboard")
print("===========================================")
print(f"User ID: {selected_user}")

result = show_recommendations(selected_user)

if result.count() > 0:
    display(result)
else:
    print("User not found. Showing available User IDs:")
    display(gold_recs.select("userId").distinct().limit(5))

# COMMAND ----------

# Calculate total users served
total_users = gold_recs.select("userId").distinct().count()

# Calculate RMSE from ALS predictions
als_predictions = spark.table("als_predictions")

rmse_df = als_predictions.withColumn(
    "squared_error",
    pow(col("rating") - col("prediction"), 2)
)

current_rmse = rmse_df.select(
    sqrt(mean("squared_error")).alias("rmse")
).collect()[0]["rmse"]

print("===========================================")
print("SYSTEM PRODUCTION SUMMARY")
print("===========================================")
print(f"Total Users Served: {total_users}")
print(f"Final ALS Model RMSE: {current_rmse:.4f}")
print("===========================================")

# COMMAND ----------

popular_recs = gold_recs.groupBy("movie_1").count() \
    .join(
        movie_titles,
        gold_recs.movie_1.cast("int") == movie_titles.movieId.cast("int"),
        "left"
    ) \
    .select(
        col("title"),
        col("count").alias("times_recommended")
    ) \
    .orderBy(col("times_recommended").desc()) \
    .limit(10)

print("Top 10 Most Recommended Movies Across the System")
display(popular_recs)