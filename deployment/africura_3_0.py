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


@st.cache(allow_output_mutation=True)
class AfricuraRecommender:
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

    def recommend_amenities(self, selected_amenity):
        # Create a dictionary to map amenities to their indices
        indices = {amen: index for index, amen in enumerate(self.clean_df['consolidated_amenities'])}

        # Check if the amenity exists in the dictionary
        if selected_amenity in indices:
            # Get the index of the amenity that matches the provided amenity
            idx = indices[selected_amenity]

            # Get the pairwise similarity scores of all amenities with that amenity
            sim_scores = list(enumerate(self.cosine_similarities[idx]))

            # Sort the amenities based on the similarity scores
            sim_scores.sort(key=lambda x: x[1], reverse=True)

            # Get the scores of the 10 most similar amenities
            sim_scores = sim_scores[1:11]

            # Get the amenity indices
            indices = [x for x, _ in sim_scores]

            return self.clean_df.set_index('consolidated_amenities').iloc[indices][
                [
                    'country',
                    'RankingType',
                    'subcategories',
                    'LowerPrice',
                    'UpperPrice',
                ]
            ]
        else:
            return "Amenity not found."

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
        recommended_places = self.clean_df.set_index('name').iloc[indices][
            [
                'country',
                'RankingType',
                'subcategories',
                'LowerPrice',
                'UpperPrice',
            ]
        ]

        return recommended_places


def load_data():
    # Load the clean_df DataFrame
    with open(r'../data/clean_df.pkl', 'rb') as f:
        clean_df = pickle.load(f)

    # Load the pickled files
    with open(r'../data/tfidfv_matrix2.pkl', 'rb') as f:
        tfidfv_matrix2 = pickle.load(f)

    with open(r'../data/cosine_sim2.pkl', 'rb') as f:
        cosine_sim2 = pickle.load(f)

    with open(r'../data/cosine_similarities.pkl', 'rb') as f:
        cosine_similarities = pickle.load(f)

    with open(r'../data/indices.pkl', 'rb') as f:
        indices = pickle.load(f)

    return clean_df, tfidfv_matrix2, cosine_sim2, cosine_similarities, indices


def main():
    st.title("Recommender System")

    # Sidebar
    option = st.sidebar.radio("Select Recommendation Type", ["Attraction", "Amenities", "Place"])

    # Add other sections using st.markdown()
    st.sidebar.subheader("About")
    st.sidebar.write("Africura is a recommendation engine that provides suggestions for locations to visit in Africa based on given preferences")

    # Set the CSS style
    st.markdown("""
        <style>
            .st-sidebar {
                width: 200px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Load the data
    clean_df, tfidfv_matrix2, cosine_sim2, cosine_similarities, indices = load_data()

    # Create the recommender object
    recommender = AfricuraRecommender(clean_df, tfidfv_matrix2, cosine_sim2, cosine_similarities, indices)

    # Get the user's preferences
    preferences = st.sidebar.multiselect("What are your preferences?", ["Hotel", "Restaurant", "Culture", "Specialty Lodging", "Bed and Breakfast","Pool", "Adventure"])

    if option == "Attraction":
        st.header("Attraction Recommendation")
        rating_threshold = st.number_input("Enter Rating Threshold", min_value=0.0, max_value=5.0, value=3.0, step=0.1)

        if st.button("Get Recommendations"):
            recommendations = recommender.recommend_attraction(rating_threshold)
            st.dataframe(recommendations)

    elif option == "Amenities":
        st.header("Amenities Recommendation")
        amenity = st.text_input("Enter Amenity Name")

        if st.button("Get Recommendations"):
            recommended_amenities = recommender.recommend_amenities(amenity)
            st.write(recommended_amenities)

    elif option == "Place":
        st.header("Place Recommendation")
        place_name = st.text_input("Enter Place Name")

        if st.button("Get Recommendations"):
            recommended_places = recommender.recommend_place(place_name)
            st.write(recommended_places)


if __name__ == "__main__":
    main()
