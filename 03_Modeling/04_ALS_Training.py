# Databricks notebook source
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# Load cleaned ratings from Student 2
ratings = spark.table("clean_ratings")

# Keep only required columns for ALS
ratings_clean = ratings.select(
    "userId",
    "movieId",
    "rating"
).dropna()

# Split data into training and testing
training, test = ratings_clean.randomSplit([0.8, 0.2], seed=42)

print("Data split: 80% Training, 20% Testing.")

# Build and train ALS model
als = ALS(
    maxIter=5,
    regParam=0.01,
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    coldStartStrategy="drop",
    seed=42
)

model = als.fit(training)

print("Model training complete!")

# Generate predictions for RMSE
predictions_df = model.transform(test)

als_predictions_df = predictions_df.select(
    "userId",
    "movieId",
    "rating",
    "prediction"
).dropna()

# Save predictions for Student 5
als_predictions_df.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("als_predictions")

print("ALS predictions saved as table: als_predictions")

display(als_predictions_df.limit(10))

# COMMAND ----------

# ================================================================
# ALS RECOMMENDATIONS — Pure Python (works on Free/Serverless tier)
# No Spark ML, No Higher-Order Functions, No Cluster needed
# ================================================================

import subprocess
subprocess.run(["pip", "install", "implicit"], capture_output=True)

import implicit
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# ─────────────────────────────────────────────────────────────────
# STEP 1: Use the existing 'ratings' DataFrame already in memory
# No table lookup needed — it's already loaded above!
# ─────────────────────────────────────────────────────────────────
ratings_pd = ratings.toPandas()

print(f"✅ Loaded {len(ratings_pd)} ratings")
print(ratings_pd.head())

# ─────────────────────────────────────────────────────────────────
# STEP 2: Encode userId and movieId into integer indexes
# ─────────────────────────────────────────────────────────────────
ratings_pd["user_idx"]  = ratings_pd["userId"].astype("category").cat.codes
ratings_pd["movie_idx"] = ratings_pd["movieId"].astype("category").cat.codes

user_map  = dict(enumerate(ratings_pd["userId"].astype("category").cat.categories))
movie_map = dict(enumerate(ratings_pd["movieId"].astype("category").cat.categories))

n_users  = ratings_pd["user_idx"].nunique()
n_movies = ratings_pd["movie_idx"].nunique()

print(f"✅ {n_users} users | {n_movies} movies")

# ─────────────────────────────────────────────────────────────────
# STEP 3: Build sparse user-item matrix
# ─────────────────────────────────────────────────────────────────
sparse_matrix = csr_matrix(
    (ratings_pd["rating"].astype(float).values,
    (ratings_pd["user_idx"].values, ratings_pd["movie_idx"].values)),
    shape=(n_users, n_movies)
)

print("✅ Sparse matrix built")

# ─────────────────────────────────────────────────────────────────
# STEP 4: Train ALS model — pure Python, zero Spark ML
# ─────────────────────────────────────────────────────────────────
als_model = implicit.als.AlternatingLeastSquares(
    factors=50,
    iterations=20,
    regularization=0.1,
    random_state=42
)
als_model.fit(sparse_matrix)

print("✅ ALS model trained")

# ─────────────────────────────────────────────────────────────────
# STEP 5: Generate top-5 recommendations for every user
# ─────────────────────────────────────────────────────────────────
rows = []
for user_idx in range(n_users):
    item_ids, scores = als_model.recommend(
        user_idx,
        sparse_matrix[user_idx],
        N=5,
        filter_already_liked_items=True
    )
    row = {"userId": int(user_map[user_idx])}
    for rank, (movie_idx, score) in enumerate(zip(item_ids, scores), start=1):
        row[f"movie_{rank}"] = int(movie_map[movie_idx])
        row[f"score_{rank}"] = float(score)
    rows.append(row)

print(f"✅ Generated recommendations for {len(rows)} users")

# ─────────────────────────────────────────────────────────────────
# STEP 6: Convert to Spark DataFrame (flat schema — no arrays/structs)
# ─────────────────────────────────────────────────────────────────
pdf = pd.DataFrame(rows)
print(pdf.head())

userRecs_flat = spark.createDataFrame(pdf)
userRecs_flat.printSchema()

# ─────────────────────────────────────────────────────────────────
# STEP 7: Find your correct catalog and schema, then save
# ─────────────────────────────────────────────────────────────────
# First run this in a SQL cell to confirm your catalog/schema:
# SELECT current_catalog(), current_schema();

target_table = "workspace.default.gold_recommendations"  # ⬅️ update if needed

userRecs_flat.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(target_table)

print(f"✅ SUCCESS: Gold Table saved → {target_table}")
display(spark.table(target_table).limit(10))