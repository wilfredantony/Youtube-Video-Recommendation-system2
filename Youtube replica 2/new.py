import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import pickle
import os

st.set_page_config(layout="wide")

# Setup the database connection
db_user = 'postgres'
db_password = 'your password'
db_host = 'your host name'
db_port = '5432'
db_name = 'youtube'
connection_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
engine = create_engine(connection_string)

# Load pre-trained models and vectorizers for different tags
def load_model(tag):
    with open(f'vector_{tag}.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open(f'pca_{tag}.pkl', 'rb') as f:
        pca = pickle.load(f)
    with open(f'model_{tag}.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    return vectorizer, pca, kmeans

# Fetch data from AWS RDS
@st.cache_data
def fetch_data():
    query = "SELECT * FROM videos2"
    df = pd.read_sql(query, engine)
    return df

df = fetch_data()

# Streamlit app
st.title("Wilfred's YouTube Replica")

# Sidebar for tag selection
tags_of_interest = ["news", "music", "sports", "technology", "gaming"]
tag = st.sidebar.selectbox('Select a Tag', [''] + tags_of_interest)  # Add an empty string option

# Load the data for the selected tag
if tag == '':
    # If no tag is selected, show all data
    tag_df = df
else:
    tag_df = df[df['tags_joined'].str.contains(tag, case=False, na=False)]

if not tag_df.empty:
    st.write(f"Displaying clustered videos for tag: {tag}" if tag else "Displaying all videos")
    
    if tag:  # Load the model only if a tag is selected
        vectorizer, pca, kmeans = load_model(tag)
        
        # Vectorize the data
        X = vectorizer.transform(tag_df['tags_joined'])
        
        # Apply PCA
        X_pca = pca.transform(X.toarray())
        
        # Predict clusters
        tag_df['cluster'] = kmeans.predict(X_pca)
    
    # Grid layout for displaying videos
    st.subheader('Video Results')
    
    num_columns = 4  # Number of columns in the grid
    rows = [tag_df.iloc[i:i + num_columns] for i in range(0, len(tag_df), num_columns)]
    
    for row in rows:
        cols = st.columns(num_columns)
        for i, video in enumerate(row.itertuples()):
            with cols[i]:
                st.image(video.thumbnail_url, use_column_width=True)
                st.markdown(f"**[{video.title}]({video.url})**")  # Clickable title
                st.write(f"Views: {video.views} | Likes: {video.likes} | Comments: {video.comments}")
else:
    st.write(f"No data available for tag: {tag}" if tag else "No videos available.")
