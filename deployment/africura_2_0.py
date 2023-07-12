# Importing necessary libraries
import pandas as pd
import numpy as np
import json
import glob
import re
import pickle
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from surprise import Dataset, Reader, KNNBasic, SVD, NMF
from surprise.model_selection import train_test_split
from surprise import accuracy as sup_accuracy
import warnings
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st


@st.cache_resource#(allow_output_mutation=True)
def load_data():
    # Load the clean_df DataFrame
    clean_df = pd.read_csv('../data/clean_data.csv')
    
    # Load the pickled files
    with open(r'../data/tfidfv_matrix2.pkl', 'rb') as f:
        tfidfv_matrix2 = pickle.load(f)
        
    with open(r'../data/.cosine_sim2.pkl', 'rb') as f:
        cosine_sim2 = pickle.load(f)

    with open(r'../data/.cosine_similarities.pkl', 'rb') as f:
        cosine_similarities = pickle.load(f)

    with open(r'../data/.indices.pkl', 'rb') as f:
        indices = pickle.load(f)

    return clean_df, tfidfv_matrix2, cosine_sim2, cosine_similarities, indices


class RecommenderSystem:
    def __init__(self, clean_df, tfidfv_matrix2, cosine_sim2, cosine_similarities, indices):
        self.clean_df = clean_df
        self.tfidfv_matrix2 = tfidfv_matrix2
        self.cosine_sim2 = cosine_sim2
        self.cosine_similarities = cosine_similarities
        self.indices = indices

    def recommend_attraction(self, rating_threshold):
        # Filter the DataFrame based on the rating threshold
        recommendations = self.clean_df[self.clean_df['rating'] > rating_threshold][['name', 'LowerPrice', 'UpperPrice','amenities', 'type', 'country']]

        # Reset the index of the recommendations DataFrame
        recommendations.reset_index(drop=True, inplace=True)

        return recommendations

    def recommend_amenities(self, query):
        # Check if the specified amenity exists in the dataset
        if query not in self.clean_df['amenities'].str.join(', '):
            st.error(f"Error: '{query}' does not exist in the dataset.")
            return None

        # Convert the string representation of amenities back into a list
        self.clean_df['amenities'] = self.clean_df['amenities'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # Get the index of the specified amenity
        indices = self.clean_df['amenities'].apply(lambda x: query in x if isinstance(x, list) else False)

        # Get the pairwise similarity scores of all items with the specified amenity
        sim_scores = self.cosine_sim2[indices]

        # Flatten the similarity scores
        sim_scores = sim_scores.flatten()

        # Get the indices of the sorted similarity scores
        indices = np.argsort(sim_scores)[::-1]

        # Get the sorted similarity scores
        sim_scores = sim_scores[indices]

        # Get the recommended items
        recommended_items = self.clean_df.iloc[indices]

        return recommended_items

    def recommend_place(self, name):
        # Create a dictionary to map place names to their indices
        indices = {title: index for index, title in enumerate(self.clean_df['name'])}

        # Check if the specified place exists in the dataset
        if name not in indices:
            st.error(f"Error: '{name}' does not exist in the dataset.")
            return None

        # Get the index of the specified place
        idx = indices[name]

        # Get the pairwise similarity scores of all places with the specified place
        sim_scores = list(enumerate(self.cosine_similarities[idx]))

        # Sort the places based on the similarity scores
        sim_scores.sort(key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar places
        sim_scores = sim_scores[1:11]

        # Get the indices of the top-N similar places
        indices = [x for x, _ in sim_scores]

        # Get the recommended places
        recommended_places = self.clean_df.iloc[indices]['name']

        return recommended_places

    def get_item_recommendations(self, item_index, top_n=5):
        # Get similarity scores for the item
        item_scores = list(enumerate(self.cosine_similarities[item_index]))

        # Sort items based on similarity scores
        item_scores = sorted(item_scores, key=lambda x: x[1], reverse=True)

        # Get top-N similar items
        top_items = item_scores[1:top_n + 1]  # Exclude the item itself

        return top_items


# Load the data
clean_df, tfidfv_matrix2, cosine_sim2, cosine_similarities, indices = load_data()

# Initialize the RecommenderSystem object
recommender = RecommenderSystem(clean_df, tfidfv_matrix2, cosine_sim2, cosine_similarities, indices)

def main():
    # Set the CSS style
    css = '''
    <style>
        /* Global styles */
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-image: url('../Data/images/ui_bg.jpg'); /* Updated path to background image */
            background-size: cover;
            background-position: center;
        }
        /* Header styles */
        header {
            background-color: white;
            color: #fff;
            padding: 20px;
            text-align: center;
        }
        h1 {
            margin: 0;
        }
        /* Navigation styles */
        nav {
            background-color: #f2f2f2;
            padding: 10px;
            text-align: center; 
        }
        ul {
            list-style-type: none;
            margin: 0;
            padding: 0;
        }
        li {
            display: inline;
            margin-right: 10px;
        }
        a {
            color: #333;
            text-decoration: none;
            padding: 5px;
        }
        /* Section styles */
        section {
            padding: 20px;
            margin-bottom: 20px;
            background-color: #f9f9f9;
            opacity: 0.9;
            border-radius: 10px;
        }
        h2 {
            margin-top: 0;
        }
    </style>
    '''
    # Render the HTML code
    st.markdown(css, unsafe_allow_html=True)
    st.markdown("<header><h1>Africura Travel Destination Recommendation System</h1></header>", unsafe_allow_html=True)

    # Add section to display suggestions
    if st.button('Get Recommendations'):
        suggestions = recommender.recommend_attraction(st.number_input('Enter Rating Threshold', min_value=0.0, max_value=5.0, value=3.0, step=0.1))
        st.dataframe(suggestions)