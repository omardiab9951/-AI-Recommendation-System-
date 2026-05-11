# 🎬 AI-Powered Movie Recommendation System

> An end-to-end collaborative filtering pipeline built on **Databricks** using the **Medallion Architecture** — from raw MovieLens data to a production-ready recommendation dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-3.x-E25A1C?style=flat&logo=apachespark&logoColor=white)
![Databricks](https://img.shields.io/badge/Databricks-Lakehouse-FF3621?style=flat&logo=databricks&logoColor=white)
![Delta Lake](https://img.shields.io/badge/Delta_Lake-Enabled-00ADD8?style=flat)
![Architecture](https://img.shields.io/badge/Architecture-Medallion-gold?style=flat)

---

## 📌 Overview

This project implements a full-scale **movie recommendation engine** using collaborative filtering (ALS) on the [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/). It demonstrates production-grade data engineering practices including layered data storage, exploratory analysis, ML modeling, and an interactive recommendation dashboard — all deployed on **Databricks**.

---

## 🏗️ Architecture

The pipeline follows the industry-standard **Medallion Architecture**, ensuring data quality and scalability at every stage.

```
┌─────────────────────────────────────────────────────────────────┐
│                     MEDALLION ARCHITECTURE                      │
│                                                                 │
│   📥 RAW          🧹 SILVER         🤖 GOLD        📊 DASHBOARD │
│  ─────────       ──────────       ─────────       ──────────── │
│  .dat files  →  Delta Tables  →  ALS Model   →  Top-N Lookup  │
│  Ingestion       Cleaning &       Training &      Production    │
│  & Storage       EDA               Scoring         Layer        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Pipeline Stages

### 1. 📥 Ingestion — Raw Layer

| Item | Detail |
|------|--------|
| **Source** | MovieLens 1M Dataset (`.dat` files) |
| **Process** | Automated ingestion into a Databricks Volume |
| **Tools** | Spark I/O, Databricks Volumes |

Raw `.dat` files are ingested and stored as-is in a Databricks Volume, preserving the original source data without transformation.

---

### 2. 🧹 Processing — Silver Layer

| Item | Detail |
|------|--------|
| **Cleaning** | Type casting, null handling, deduplication |
| **Analysis** | EDA on user behavior and rating distributions |
| **Storage** | Persisted as Delta tables for high-performance querying |

Data is cleaned, validated, and enriched with schema enforcement before being written to Silver Delta tables — the single source of truth for all downstream processes.

---

### 3. 🤖 Modeling — Gold Layer

| Item | Detail |
|------|--------|
| **Algorithm** | Collaborative Filtering via ALS (Alternating Least Squares) |
| **Evaluation** | Baseline Average Model with RMSE via pure Spark SQL |
| **Segmentation** | Users classified as Enthusiasts, Power Users, and Casuals |

A custom ALS implementation generates personalized top-5 recommendations per user. User segmentation applies statistical thresholds to profile audience types for targeted recommendations.

---

### 4. 📊 Production — Dashboard Layer

| Item | Detail |
|------|--------|
| **Feature** | Real-time Top-N recommendation lookup |
| **Design** | Decoupled from model training for maximum uptime |
| **Output** | Interactive notebook dashboard for business use |

The production layer serves pre-computed Gold table recommendations through a fast, decoupled lookup system — ensuring zero dependency on model retraining for live queries.

---

## 📂 Repository Structure

```
📦 movie-recommendation-system/
│
├── 📁 01_Ingestion/
│   └── 01_Cloud_Setup.ipynb              # Environment & Volume configuration
│
├── 📁 02_Processing/
│   ├── 02_Data_Cleaning.ipynb            # Schema enforcement & Silver tables
│   └── 03_Exploratory_Analysis.ipynb     # Data visualization & EDA
│
├── 📁 03_Modeling/
│   ├── 04_ALS_Training.ipynb             # Recommendation engine & ALS logic
│   └── 05_User_Segmentation.ipynb        # Accuracy metrics & user profiling
│
└── 📁 04_Production/
    └── 06_Final_Dashboard.ipynb          # Interactive recommendation UI
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Platform | Databricks (Serverless / Unity Catalog) |
| Language | Python 3.10+, PySpark, Spark SQL |
| Storage | Delta Lake, Databricks Volumes |
| ML | ALS Collaborative Filtering (`implicit` library) |
| Data | MovieLens 1M — 1,000,209 ratings, 6,040 users, 3,706 movies |
| Architecture | Medallion (Bronze → Silver → Gold) |

---

## 🚀 Getting Started

### Prerequisites

- Databricks workspace (Free Edition or above)
- Unity Catalog enabled
- Python 3.10+

### Run Order

Execute notebooks **in sequence** — each stage depends on the output of the previous one:

```
01_Cloud_Setup  →  02_Data_Cleaning  →  03_EDA  →  04_ALS_Training  →  05_Segmentation  →  06_Dashboard
```

### Install Dependencies

```python
%pip install implicit scipy
```

---

## 📊 Dataset

**MovieLens 1M** — collected by the GroupLens Research Project at the University of Minnesota.

| File | Records | Description |
|------|---------|-------------|
| `ratings.dat` | 1,000,209 | User–movie ratings (1–5 stars) |
| `movies.dat` | 3,706 | Movie titles and genres |
| `users.dat` | 6,040 | User demographics |

---

## 📈 Key Results

- ✅ Full Medallion pipeline from raw `.dat` files to Gold Delta tables
- ✅ Top-5 personalized recommendations generated for all users
- ✅ User segments profiled: **Enthusiasts**, **Power Users**, **Casuals**
- ✅ RMSE baseline computed for model evaluation
- ✅ Production dashboard decoupled from training pipeline

---

## 👤 Author

Built as a data engineering & ML project on **Databricks**, demonstrating end-to-end pipeline design with real-world datasets and production-ready architecture patterns.

---

*MovieLens data provided by [GroupLens Research](https://grouplens.org/datasets/movielens/).*
