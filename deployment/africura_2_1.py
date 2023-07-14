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
from ipywidgets import interact_manual
from IPython.display import display, HTML
from ipywidgets import Dropdown

@st.cache_resource
def load_data():
    # Load the clean_df DataFrame
    with open(r'../data/clean_df.pkl', 'rb') as f:
        clean_df = pickle.load(f)

    # Load the pickled files
    with open(r'../data/tfidf_matrix2.pkl', 'rb') as f:
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
        
    def get_item_recommendations(item_index, cosine_similarities, top_n=5):
        # Get similarity scores for the item
        item_scores = list(enumerate(cosine_similarities[item_index]))

        # Sort items based on similarity scores
        item_scores = sorted(item_scores, key=lambda x: x[1], reverse=True)

        # Get top-N similar items
        top_items = item_scores[1 : top_n + 1]  # Exclude the item itself

        return top_items

        # Get recommendations for a specific item (e.g., item with index 0)
        item_index = 0
        recommendations_a = get_item_recommendations(item_index, cosine_similarities)

        # Print the top 5 recommendations
        #for item_id, similarity in recommendations:
        #print(f"Item ID: {item_id}, Similarity: {similarity}")


    def recommend_attraction(self, rating_threshold):
        # Filter the DataFrame based on the rating threshold
        recommendations = self.clean_df[self.clean_df['rating'] > rating_threshold][['name', 'LowerPrice', 'UpperPrice','amenities', 'type', 'country']]

        # Reset the index of the recommendations DataFrame
        recommendations.reset_index(drop=True, inplace=True)

        return recommendations
        
    def recommend_amenities(self, selected_amenity, cosine_sim2, clean_df):
        # Create a dictionary to map amenities to their indices
        indices = {amen: index for index, amen in enumerate(self.clean_df['consolidated_amenities'])}

        # Check if the amenity exists in the dictionary
        if selected_amenity in indices:
            # Get the index of the amenity that matches the provided amenity
            idx = indices[selected_amenity]

            # Get the pairwise similarity scores of all amenities with that amenity
            sim_scores = list(enumerate(self.cosine_sim2[idx]))

            # Sort the amenities based on the similarity scores
            sim_scores.sort(key=lambda x: x[1], reverse=True)

            # Get the scores of the 10 most similar amenities
            sim_scores = sim_scores[1:11]

            # Get the amenity indices
            indices = [x for x, _ in sim_scores]

            return self.clean_df.set_index('consolidated_amenities').iloc[indices][
                [
                    'name',
                    'country',
                    'RankingType',
                    'amenities',
                    'LowerPrice',
                    'UpperPrice',
                ]
            ]
        else:
            return "Amenity not found."
    
            
    def get_recommended_amenities(amenity):
        recommended_amenities = recommend_amenities(amenity, cosine_sim2, clean_df)
        if isinstance(recommended_amenities, str):
            display(HTML(recommended_amenities))
        else:
            display(recommended_amenities)

 
    def recommend_place(self, name, cosine_sim2, clean_df):
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


        
    def get_item_recommendations(self, preference, top_n=5):
     # Get the index of the preference
        preference_index = self.clean_df['consolidated_amenities'].tolist().index(preference)

        # Get similarity scores for the preference
        preference_scores = list(enumerate(self.cosine_similarities[preference_index]))

        # Sort preferences based on similarity scores
        preference_scores = sorted(preference_scores, key=lambda x: x[1], reverse=True)

        # Get top-N similar preferences
        top_preferences = preference_scores[1:top_n + 1]  # Exclude the preference itself

        return top_preferences


def main():

    st.set_page_config(layout="wide")
    menu = ['About', 'Recomenders']
    selection = st.sidebar.selectbox("Select Menu", menu)
    
    # Sidebar
    option = st.sidebar.radio("Select Recommendation Type", ["Attraction", "Amenities", "Place", "Preferences"])
        
    # Add other sections using st.markdown()
    st.sidebar.subheader("About")
    st.sidebar.write("Africura is a recommendation engine that provides suggestions for locations to visit in Africa based on given preferences")
    
    if selection == "About" :
        st.markdown("## Welcome to Africura!")
        st.markdown("A one stop shop for all you WanderLusters. ")
         
        with st.markdown("## Contact"):
            with st.form(key='contact-form'):
                st.markdown("Any queries? Please fill out the form below and we will get back to you as soon as possible.")
                st.markdown("### Message")
                message = st.text_area(label='Enter your message here')
                st.markdown("### Contact Information")
                name = st.text_input(label='Name')
                email = st.text_input(label='Email')
                phone = st.text_input(label='Phone')
                st.markdown("###")
                submit_button = st.form_submit_button(label='Submit')
                if submit_button:
                    st.markdown("Thank you for getting in touch. We will get back to you as soon as possible.")
        st.markdown("<footer><p>&copy; 2023 Africura Travel Destination Recommendation System. All rights reserved.</p></footer>", unsafe_allow_html=True)   
        
        
        
        
        
    if selection == "Recomenders" :    
        st.title("Recommender System")
    
        
        # Add other sections using st.markdown()
        #st.sidebar.subheader("About")
        #st.sidebar.write("Africura is a recommendation engine that provides suggestions for locations to visit in Africa based on given preferences")

        # Set the CSS style
        st.markdown("""
            <style>
                .st-sidebar {
                    width: 100px;
                }
            </style>
        """, unsafe_allow_html=True)

        # Get the user's preferences
        preferences = st.sidebar.multiselect("What are your preferences?", ["Hotel", "Restaurant", "Culture", "Specialty Lodging", "Bed and Breakfast","Pool", "Adventure"])

        # Load the data
        clean_df, tfidfv_matrix2, cosine_sim2, cosine_similarities, indices = load_data()

        # Create the RecommenderSystem object
        recommender = RecommenderSystem(clean_df, tfidfv_matrix2, cosine_sim2, cosine_similarities, indices)

        # Recommend locations based on the user's preferences
        if option == "Attraction":
            rating_threshold = st.number_input("Enter Rating Threshold", min_value=0.0, max_value=5.0, value=3.0, step=0.1)

            if st.button("Get Recommendations"):
                recommendations = recommender.recommend_attraction(rating_threshold)
                st.dataframe(recommendations)

        elif option == "Amenities":      
            amenity = st.text_input("Enter Amenity Name")

            if st.button("Get Recommendations"):
                recommended_amenities = recommender.recommend_amenities(amenity, cosine_sim2, clean_df)
                st.write(recommended_amenities)
        elif option == "Preferences":
            amenities_dropdown = st.selectbox("Consolidated Amenities", clean_df['consolidated_amenities'].unique())                         
            def get_recommended_amenities(amenity):
                recommended_amenities = recommend_amenities(amenity, cosine_sim2, clean_df)
                if isinstance(recommended_amenities, str):
                    display(HTML(recommended_amenities))
                else:
                    display(recommended_amenities)
                st.dataframe(get_recommended_amenities(amenities_dropdown))
                #amenities_dropdown = st.selectbox("Select Amenity:", clean_df['consolidated_amenities'])
                


            # Create a dropdown menu with the available amenities
                #amenities_dropdown = Dropdown(options=clean_df['consolidated_amenities'].unique(), description='Select Amenity:')
        
        
        elif option == "Place":
 
            st.header("Place Recommendation")
            place_name = st.text_input("Enter Place Name")

            if st.button("Get Recommendations"):
                recommended_places = recommender.recommend_place(place_name, cosine_sim2, clean_df)
                st.write(recommended_places)
         

        # Additional sections for the amenities tab
 

    
    
if __name__ == "__main__":
    main()