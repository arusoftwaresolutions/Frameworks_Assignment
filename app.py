# Part 4: Streamlit App

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("metadata.csv", low_memory=False)
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
    df['year'] = df['publish_time'].dt.year
    df['abstract_word_count'] = df['abstract'].fillna("").apply(lambda x: len(x.split()))
    return df.dropna(subset=['title','publish_time'])

df = load_data()

# Title
st.title("CORD-19 Data Explorer")
st.write("Interactive exploration of COVID-19 research metadata")

# Year filter
years = st.slider("Select publication year range", int(df['year'].min()), int(df['year'].max()), (2020,2021))
df_filtered = df[(df['year'] >= years[0]) & (df['year'] <= years[1])]

st.write(f"Showing {df_filtered.shape[0]} papers between {years[0]} and {years[1]}")

# Publications by year
year_counts = df_filtered['year'].value_counts().sort_index()
fig, ax = plt.subplots()
ax.bar(year_counts.index, year_counts.values)
ax.set_title("Publications by Year")
st.pyplot(fig)

# Top Journals
top_journals = df_filtered['journal'].value_counts().head(10)
fig, ax = plt.subplots()
sns.barplot(x=top_journals.values, y=top_journals.index, ax=ax)
ax.set_title("Top Journals")
st.pyplot(fig)

# Word Cloud
title_words = " ".join(df_filtered['title'].dropna().astype(str).tolist())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(title_words)
fig, ax = plt.subplots(figsize=(12,6))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

# Show sample data
st.subheader("Sample of Data")
st.write(df_filtered[['title','authors','journal','year']].head(20))
