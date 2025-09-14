import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import pandas as pd


# Part 1: Data Loading and Basic Exploration

# Load metadata.csv 
try:
    df = pd.read_csv("metadata.csv", low_memory=False)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: metadata.csv not found. Please download and place in working directory.")

# First few rows
print("\nFirst 5 rows:")
print(df.head())

# Shape and info
print("\nShape of dataset:", df.shape)
print("\nInfo:")
print(df.info())

# Missing values
print("\nMissing values per column:")
print(df.isnull().sum().sort_values(ascending=False).head(15))

# Basic stats
print("\nBasic statistics for numerical columns:")
print(df.describe())
# Part 2: Data Cleaning and Preparation

# Convert publish_time to datetime
df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')

# Extract year
df['year'] = df['publish_time'].dt.year

# Create abstract word count (if abstract exists)
df['abstract_word_count'] = df['abstract'].fillna("").apply(lambda x: len(x.split()))

# Drop rows with missing titles or publish_time (important for analysis)
df_clean = df.dropna(subset=['title', 'publish_time']).copy()

print("\nAfter cleaning:")
print(df_clean.shape)
print(df_clean[['title', 'year', 'abstract_word_count']].head())

# Part 3: Data Analysis and Visualization
# Publications by year
year_counts = df_clean['year'].value_counts().sort_index()

plt.figure(figsize=(10,5))
plt.bar(year_counts.index, year_counts.values)
plt.title("Publications by Year")
plt.xlabel("Year")
plt.ylabel("Number of Papers")
plt.show()

# Top Journals
top_journals = df_clean['journal'].value_counts().head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=top_journals.values, y=top_journals.index)
plt.title("Top 10 Journals Publishing COVID-19 Research")
plt.xlabel("Number of Papers")
plt.ylabel("Journal")
plt.show()

# Word Cloud of Titles
title_words = " ".join(df_clean['title'].dropna().astype(str).tolist())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(title_words)

plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Frequent Words in Paper Titles")
plt.show()

# Distribution of papers by source
top_sources = df_clean['source_x'].value_counts().head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=top_sources.values, y=top_sources.index)
plt.title("Distribution of Papers by Source (Top 10)")
plt.xlabel("Number of Papers")
plt.ylabel("Source")
plt.show()
