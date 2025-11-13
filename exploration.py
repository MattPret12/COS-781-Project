import pandas as pd
from pathlib import Path
import numpy as np

DATA_DIR = Path("dataset")

business_path = DATA_DIR / "yelp_academic_dataset_business.json"
review_path   = DATA_DIR / "yelp_academic_dataset_review.json"
user_path     = DATA_DIR / "yelp_academic_dataset_user.json"

# 1. Load line-delimited JSON
business_df = pd.read_json(business_path, lines=True)
review_df   = pd.read_json(review_path, lines=True)
user_df     = pd.read_json(user_path, lines=True)

print("Business shape:", business_df.shape)
print("Review shape  :", review_df.shape)
print("User shape    :", user_df.shape)

print("\nBusiness columns:\n", business_df.columns)
print("\nReview columns:\n", review_df.columns)
print("\nUser columns:\n", user_df.columns)

# Examples: business
print(business_df.head())
print(business_df.isna().mean().sort_values(ascending=False))  # missingness
print(business_df[['stars', 'review_count']].describe())

# Examples: user
print(user_df[['review_count', 'fans', 'average_stars']].describe())
print(user_df['yelping_since'].head())

# Examples: review
print(review_df[['stars', 'useful', 'funny', 'cool']].describe())
print(review_df['date'].head())

print("=== BUSINESS TABLE ===")
print(business_df.head())            # first 5 rows
print(business_df.info())            # column dtypes and nulls
print(business_df.describe())        # numeric columns summary
print("\nMissing values (%):")
print(business_df.isna().mean().sort_values(ascending=False) * 100)

print("=== REVIEW TABLE ===")
print(review_df.head())
print(review_df.info())
print(review_df.describe())
print("\nMissing values (%):")
print(review_df.isna().mean().sort_values(ascending=False) * 100)

print("=== USER TABLE ===")
print(user_df.head())
print(user_df.info())
print(user_df.describe())
print("\nMissing values (%):")
print(user_df.isna().mean().sort_values(ascending=False) * 100)

import matplotlib.pyplot as plt

# The dataset is VERY large. Plotting the full review dataset may freeze your machine.
# Take small samples to avoid RAM issues
business_sample = business_df.sample(5000, random_state=42)
user_sample     = user_df.sample(5000, random_state=42)
review_sample   = review_df.sample(5000, random_state=42)

plt.figure(figsize=(6,4))
business_sample['stars'].hist(bins=10)
plt.title("Distribution of Business Star Ratings")
plt.xlabel("Stars")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6,4))
business_sample['review_count'].apply(lambda x: max(x,1)).apply(np.log10).hist(bins=20)
plt.title("Business Review Count Distribution (log10 scale)")
plt.xlabel("log10(review_count)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6,4))
review_sample['stars'].hist(bins=10)
plt.title("Distribution of Review Star Ratings")
plt.xlabel("Stars")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6,4))
review_sample['text'].str.len().hist(bins=20)
plt.title("Distribution of Review Text Length")
plt.xlabel("Character Count")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6,4))
user_sample['average_stars'].hist(bins=10)
plt.title("Distribution of User Average Stars")
plt.xlabel("Average Stars")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6,4))
user_sample['review_count'].apply(lambda x: max(x,1)).apply(np.log10).hist(bins=20)
plt.title("User Review Count Distribution (log10 scale)")
plt.xlabel("log10(review_count)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6,4))
user_sample['fans'].hist(bins=20)
plt.title("Distribution of User Fans")
plt.xlabel("Fans")
plt.ylabel("Count")
plt.show()

import seaborn as sns

numeric_cols = ['review_count','useful','funny','cool','fans','average_stars']

plt.figure(figsize=(8,6))
sns.heatmap(user_sample[numeric_cols].corr(), annot=True, cmap='Blues')
plt.title("Correlation Heatmap of User Metrics")
plt.show()

