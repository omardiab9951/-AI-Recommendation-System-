# Task 5: Specialist & Evaluator

## Objective
This task focuses on completing the recommendation system by adding user segmentation and model evaluation.

The implementation will use Apache Spark and Spark MLlib inside Databricks.

## Task Scope
Student 5 is responsible for two main parts:
- User Segmentation using K-Means Clustering.
- Model Evaluation using RMSE.

This task does not cover the full data pipeline, dashboard, or final production deployment.

## Dataset
The current dataset used for development is MovieLens 25M.

The dataset was loaded successfully using Spark on Databricks.

Verified dataset status:
- ratings.csv contains 25,000,095 ratings.
- movies.csv was loaded successfully.
- ratings_df columns are: userId, movieId, rating, timestamp.
- movies_df columns are: movieId, title, genres.

## Data Location
The dataset is stored in Databricks storage, not inside GitHub.

Current expected data paths:
- /Volumes/workspace/default/ml25m/ratings.csv
- /Volumes/workspace/default/ml25m/movies.csv

Large data files must not be pushed to GitHub.

## Tools and Technology
This task will use:
- Databricks
- Apache Spark
- PySpark DataFrames
- Spark MLlib

This task will not use:
- Pandas
- scikit-learn
- local Python processing

## Planned Workflow

### Phase 1: Data Loading Check
Load ratings.csv and movies.csv using Spark.

Goal:
- Confirm that Spark can read the dataset.
- Check the schema.
- Confirm the number of ratings.

Current status:
- Completed.

### Phase 2: Data Preparation
Prepare the working ratings DataFrame for Task 5.

Required columns:
- userId
- movieId
- rating

Planned checks:
- Cast userId to integer.
- Cast movieId to integer.
- Cast rating to float or double.
- Remove null values from required columns.

### Phase 3: User Feature Engineering
Create user-level features from ratings data.

Planned features:
- avg_rating: average rating given by each user.
- rating_count: number of ratings made by each user.

Purpose:
These features will represent user behavior and will be used for K-Means Clustering.

### Phase 4: K-Means User Segmentation
Apply K-Means Clustering using Spark MLlib.

Initial plan:
- Use avg_rating and rating_count as input features.
- Start with k = 3 clusters.
- Analyze the result before changing k.

Expected output:
- userId
- avg_rating
- rating_count
- cluster

### Phase 5: Cluster Analysis
Analyze the generated user clusters.

Required summary:
- Number of users in each cluster.
- Average rating per cluster.
- Average rating count per cluster.

Goal:
Explain what each cluster represents.

### Phase 6: RMSE Evaluation
Calculate RMSE to evaluate recommendation model accuracy.

Expected prediction columns:
- userId
- movieId
- rating
- prediction

If Student 4 predictions are available:
- Load the predictions directly.
- Calculate RMSE.

If Student 4 predictions are not available:
- Build a temporary ALS baseline only to generate predictions.
- Clearly mark it as temporary.
- Replace it later with Student 4 final predictions.

### Phase 7: Final Summary
Write a short final interpretation.

The final summary should include:
- Dataset used.
- Number of ratings.
- Number of user clusters.
- Cluster interpretation.
- RMSE value.
- Whether the model needs tuning.

## Final Deliverables
Student 5 final deliverables:
- docs/task5_plan.md
- 03_Modeling/05_User_Segmentation.py
- Cluster summary output.
- RMSE output.
- Short explanation for the final report or presentation.

## Notes
The current work is based on MovieLens 25M because it is large enough for Spark and suitable for recommendation systems.

The data has already been verified in Databricks:
- ratings_df loaded successfully.
- movies_df loaded successfully.
- Number of ratings: 25,000,095.