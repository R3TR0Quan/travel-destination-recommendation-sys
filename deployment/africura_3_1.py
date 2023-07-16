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
from PIL import Image

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

st.set_page_config(layout="wide")
class RecommendationEngine:
    def __init__(self, cosine_similarities, cosine_sim2, clean_df):
        self.cosine_similarities = cosine_similarities
        self.cosine_sim2 = cosine_sim2
        self.clean_df = clean_df

        
    def get_item_recommendations(self, item_index, cosine_similarities, top_n=5):
        # Get similarity scores for the item
        item_scores = list(enumerate(cosine_similarities[item_index]))

        # Sort items based on similarity scores
        item_scores = sorted(item_scores, key=lambda x: x[1], reverse=True)

        return item_scores[1 : top_n + 1]

    def recommend_attraction(self, rating_threshold):
        recommendations = self.clean_df[self.clean_df['rating'] == rating_threshold][['name', 'LowerPrice', 'UpperPrice', 'amenities', 'type', 'country']]
        recommendations.reset_index(drop=True, inplace=True)
        return recommendations
    
    def recommend_amenities(self, selected_amenity, cosine_sim2, clean_df):
        # Create a dictionary to map amenities to their indices
        indices = {amen: index for index, amen in enumerate(self.clean_df['combined_amenities'])}

    def recommend_amenities(self, combined_amenities):
        indices = {title: index for index, title in enumerate(self.clean_df['combined_amenities'])}
        idx = indices[combined_amenities]
        sim_scores = list(enumerate(np.dot(self.cosine_sim2[idx], self.cosine_similarities)))
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        indices = [x for x, _ in sim_scores]
        return self.clean_df.set_index('combined_amenities').iloc[indices][
            [
                'name',
                'country',
                'RankingType',
                'subcategories',
                'LowerPrice',
                'UpperPrice',
            ]
        ]

            
    def get_recommended_amenities(self, amenity):
        recommended_amenities = self.recommend_amenities(amenity, cosine_sim2, clean_df)
        if isinstance(recommended_amenities, str):
            display(HTML(recommended_amenities))
        else:
                display(recommended_amenities)
                
    #def get_country_amenities(self, clean_df):
       # return (
         #   self.clean_df.groupby('country')['consolidated_amenities']
         #   .apply(list)
          #  .reset_index()
      #  )

    def recommend_country(self, country):
        indices = {title: index for index, title in enumerate(self.clean_df['country'])}
        idx = indices[country]
        sim_scores = list(enumerate(np.dot(self.cosine_sim2[idx], self.cosine_similarities)))
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:1000]
        indices = [x for x, _ in sim_scores]
        recommended_country = self.clean_df.set_index('country').iloc[indices][['name', 'city', 'RankingType', 'subcategories', 'LowerPrice', 'UpperPrice']]

        filtered_recommendations = recommended_country[recommended_country.index == country]
        return pd.DataFrame(filtered_recommendations)

    def recommend_place(self, name):
        indices = {title: index for index, title in enumerate(self.clean_df['name'])}
        idx = indices[name]
        sim_scores = list(enumerate(np.dot(self.cosine_sim2[idx], self.cosine_similarities)))
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        indices = [x for x, _ in sim_scores]
        return self.clean_df.set_index('name').iloc[indices][
            ['country', 'RankingType', 'subcategories', 'LowerPrice', 'UpperPrice']
        ]


# Load the data
clean_df, tfidfv_matrix2, cosine_sim2, cosine_similarities, indices = load_data()

# Create the RecommenderSystem object
recommender = RecommendationEngine(clean_df, cosine_sim2, cosine_similarities)


def main():
    # Set the CSS style
    st.markdown("""
        <style>
            body {
                background-image: Image.open("../Data/images/ui_bg.jpg");
                background-size: cover;
                
            }
        </style>
    """, unsafe_allow_html=True)



    menu = ['About', 'Recommenders']
    selection = st.sidebar.selectbox("Select Menu", menu)



    # Add other sections using st.markdown()
    st.sidebar.subheader("About")
    st.sidebar.write("Africura is a recommendation engine that provides suggestions for locations to visit in Africa based on given preferences")

    if selection == "About":
        _extracted_from_main_25()
    if selection == "Recommenders":
        _extracted_from_main_89()     


# TODO Rename this here and in `main`
def _extracted_from_main_89():
    #st.title("Recommender System")    
    #st.markdown("Africura is a recommendation engine that provides suggestions for locations to visit in Africa based on given preferences")

    st.markdown("## Recommendation Section")

    rating_threshold = st.number_input("Enter Rating Threshold", min_value=0.0, max_value=5.0, value=3.0, step=0.1)

    if st.button("Get Recommendations rating"):
        recommendations = recommender.recommend_attraction(rating_threshold)
        st.dataframe(recommendations)

    place_name = st.text_input("Enter Place Name")

    if st.button("Get Recommendations place"):
        recommended_places = recommender.recommend_place(place_name, cosine_sim2, clean_df)
        st.write(recommended_places)

    #get_recommended_amenities(amenities_dropdown)

    amenities = clean_df['combined_amenities'].str.split(', ').explode().unique()

    if len(amenities) >= 0:
        # Convert the amenities array to a list
        amenities_list = amenities.tolist()

        if selected_amenities := st.multiselect(
            "Select Amenities", amenities_list
        ):
            # Filter the clean_df DataFrame based on the selected amenities
            filtered_df = clean_df[clean_df['combined_amenities'].str.contains('|'.join(selected_amenities))]

            # Get the recommendations for the selected amenities
            recommendations = recommender.recommend_amenities(selected_amenities[0], cosine_sim2, filtered_df)

            # Display the recommendations
            st.write(recommendations)
        else:
            st.write("Please select at least one amenity.")
    else:
        st.write("No amenities found.")



    amenities_data = recommender.recommend_country(clean_df)

    country = st.selectbox("Select Country", amenities_data['country'])

    if st.button("Get Amenities"):
        selected_amenities = amenities_data.loc[amenities_data['country'] == country, 'combined_amenities'].iloc[0]

        if selected_amenities := st.multiselect(
            "Select Amenities", selected_amenities
        ):
            # Filter the clean_df DataFrame based on the selected amenities and country
            filtered_df = clean_df[
                (clean_df['combined_amenities'].str.contains('|'.join(selected_amenities))) &
                (clean_df['country'] == country)
            ]

            # Get the recommendations for the selected amenities and country
            recommendations = recommender.recommend_amenities(filtered_df, cosine_sim2)

            # Display the recommendations
            st.write(recommendations)
        else:
            st.write("Please select at least one amenity.")     


# TODO Rename this here and in `main`
def _extracted_from_main_25():
    st.markdown("## Welcome to Africura!")
    st.markdown("A one-stop-shop for all you WanderLusters. ")
    st.write("Personalized Destination Recommendations: Utilize your recommendation system to suggest personalized travel destinations to users based on their budget constraints. Take into account factors such as customer reviews, location preferences, amenities, and residence types to offer tailored recommendations that align with their preferences.")
    st.write("Top Destinations in Africa: Analyze the data from your system to identify the top tourist destinations in Africa based on customer ratings, popularity, and positive reviews. Highlight these destinations to attract users and showcase the most sought-after locations.")
    st.write("Customer Loyalty and Engagement: Leverage your recommendation system to foster customer loyalty and encourage repeat customers. Provide incentives or rewards for users who book multiple trips or engage with your platform frequently. Offer personalized promotions or discounts for their preferred destinations to enhance customer engagement and satisfaction. Continuous Improvement: Collect user information and feedback to improve the recommendations in the long run. Implement mechanisms to gather user reviews and ratings for destinations they have visited through your system. Utilize this feedback to refine your recommendation algorithms, enhance the accuracy of predictions, and provide even better suggestions to future users.")
    map_data = {
        'layout': go.Layout(
            title='Places to visit by Location',
            autosize=True,
            hovermode='closest',
            mapbox=dict(
                style='stamen-terrain',
                bearing=0,
                center=dict(lat=8, lon=20),
                pitch=0,
                zoom=2,
            ),
        ),
        'data': go.Scattermapbox(
            lat=clean_df['latitude'],
            lon=clean_df['longitude'],
            mode='markers',
            marker=dict(size=5, color=clean_df['rating'], opacity=0.8),
            text=[f'Price: ${i}' for i in clean_df['UpperPrice']],
            hovertext=clean_df.apply(
                lambda x: f"Ranking Type: ${x['RankingType']}, Location: {x['locationString']}",
                axis=1,
            ),
        ),
    }

    # Create the map figure
    fig = go.Figure(data=[map_data['data']], layout=map_data['layout'])

    # Show the map figure in Streamlit
    st.plotly_chart(fig, width='100%')

    with st.markdown("## Contact"):
        with st.form(key='contact-form'):
            _extracted_from_main_()
    st.markdown("<footer><p>&copy; 2023 Africura Travel Destination Recommendation System. All rights reserved.</p></footer>", unsafe_allow_html=True)     



def _extracted_from_main_():
    st.markdown("Any queries? Please fill out the form below and we will get back to you as soon as possible.")
    st.markdown("### Message")
    message = st.text_area(label='Enter your message here')
    st.markdown("### Contact Information")
    name = st.text_input(label='Name')
    email = st.text_input(label='Email')
    phone = st.text_input(label='Phone')
    st.markdown("###")
    if submit_button := st.form_submit_button(label='Submit'):
        st.markdown("Thank you for getting in touch. We will get back to you as soon as possible.")     
        
            
if __name__ == "__main__":
    main()
