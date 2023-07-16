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
    def __init__(self, clean_df, cosine_sim2, cosine_similarities):
        self.clean_df = clean_df
        self.cosine_sim2 = cosine_sim2
        self.cosine_similarities = cosine_similarities

    def recommend_place(self, name, cosine_similarities, cosine_sim2):
        indices = {title: index for index, title in enumerate(self.clean_df['name'])}
        idx = indices[name]
        sim_scores = list(enumerate(np.dot(self.cosine_sim2[idx], self.cosine_similarities)))
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        indices = [x for x, _ in sim_scores]
        return self.clean_df.set_index('name').iloc[indices][
            ['country', 'RankingType', 'subcategories', 'LowerPrice', 'UpperPrice']
        ]

    def select_amenities(self, combined_amenities, cosine_similarities, cosine_sim2):
        indices = {title: index for index, title in enumerate(self.clean_df['combined_amenities'])}

        # Check if the combined_amenities exists in the indices dictionary
        if combined_amenities in indices:
            idx = indices[combined_amenities]
            sim_scores = list(enumerate(np.dot(self.cosine_sim2[idx], self.cosine_similarities)))
            sim_scores.sort(key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:11]
            indices = [x for x, _ in sim_scores]
            return self.clean_df.set_index('combined_amenities').iloc[indices][['name', 'country', 'RankingType', 'subcategories', 'LowerPrice', 'UpperPrice']]
        else:
            return f"No recommendations found for {combined_amenities}."


    def recommend_amenities(self, amenities, cosine_similarities, cosine_sim2):
        indices = {title: index for index, title in enumerate(self.clean_df['amenities'])}
        idx = indices[amenities]
        sim_scores = list(enumerate(np.dot(self.cosine_sim2[idx], self.cosine_similarities)))
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        indices = [x for x, _ in sim_scores]
        return self.clean_df.set_index('amenities').iloc[indices][
            ['name', 'country', 'RankingType', 'subcategories', 'LowerPrice', 'UpperPrice']
        ]
        
    
    def recommend_subcategory(self, subcategories, cosine_similarities, cosine_sim2):
        indices = {title: index for index, title in enumerate(self.clean_df['subcategories'])}
        idx = indices[subcategories]
        sim_scores = list(enumerate(np.dot(self.cosine_sim2[idx], self.cosine_similarities)))
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        indices = [x for x, _ in sim_scores]
        return self.clean_df.set_index('subcategories').iloc[indices][
            ['name', 'country', 'RankingType', 'type', 'LowerPrice', 'UpperPrice']
        ]    

    def recommend_attraction(self, rating_threshold, cosine_similarities, cosine_sim2):
        recommendations = self.clean_df[self.clean_df['rating'] < rating_threshold][
            ['name', 'LowerPrice', 'UpperPrice', 'amenities', 'type', 'country']
        ]
        recommendations.reset_index(drop=True, inplace=True)
        return recommendations

    def recommend_country(self, country, cosine_similarities, cosine_sim2):
        indices = {title: index for index, title in enumerate(self.clean_df['country'])}
        idx = indices[country]
        sim_scores = list(enumerate(np.dot(self.cosine_sim2[idx], self.cosine_similarities)))
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:1000]
        indices = [x for x, _ in sim_scores]
        recommended_country = self.clean_df.set_index('country').iloc[indices][
            ['name', 'city', 'RankingType', 'subcategories', 'LowerPrice', 'UpperPrice']
        ]

        filtered_recommendations = recommended_country[recommended_country.index == country]
        return pd.DataFrame(filtered_recommendations)
def create_account():
    """Creates a new account."""

    username = st.text_input("Username")
    password = st.text_input("Password")

    if st.button("Create Account"):
        if len(username) > 0 and len(password) > 0:
            st.success("Account created successfully!")
        else:
            st.error("Please enter a username and password.")
            
    # Load the data
clean_df, tfidfv_matrix2, cosine_sim2, cosine_similarities, indices = load_data()

    # Create the RecommenderSystem object
recommender = RecommendationEngine(clean_df, cosine_sim2, cosine_similarities)


def main():
    menu = ['About', 'Join' ,'Recomenders', 'Gallery']
    
    selection = st.sidebar.selectbox("Select Menu", menu)
    
    #if selection == "About":
       # _extracted_from_main_25()
    #if selection == "Recommenders":
       # _extracted_from_main_89()  
        
    # Set the CSS style
    st.markdown(
        """<style>
            body {
                background-image: Image.open("../Data/images/ui_bg.jpg");
                background-size: cover;
                
            }
        </style>"""
    , unsafe_allow_html=True)
    



   # Add other sections using st.markdown()
    st.sidebar.subheader("About")
    st.sidebar.write("Africura is a recommendation engine that provides suggestions for locations to visit in Africa based on given preferences")

                
    if selection == "Join":
        create_account()
            
    
    # Sidebar
    #option = st.sidebar.radio("Select Recommendation Type", ["Attraction", "Place", "Preferences", "Amenities"])
   
#def _extracted_from_main_25():
    if selection == "About" :
        with st.sidebar.expander(""):
            pass
        st.markdown("# Welcome to Africura!")
        st.markdown("##### A one stop shop for all you WanderLusters. ")
        st.write("## Africa Travel Recommendation System")
        
        
        st.write("This recommendation system uses machine learning to recommend places to visit in Africa. The system takes into account a user's interests, budget, and travel dates to generate personalized recommendations.")
        st.write("Top Destinations in Africa: Analyze the data from your system to identify the top tourist destinations in Africa based on customer ratings, popularity, and positive reviews. Highlight these destinations to attract users and showcase the most sought-after locations.") 
        st.write("Customer Loyalty and Engagement: Leverage your recommendation system to foster customer loyalty and encourage repeat customers. Provide incentives or rewards for users who book multiple trips or engage with your platform frequently. Offer personalized promotions or discounts for their preferred destinations to enhance customer engagement and satisfaction. Continuous Improvement: Collect user information and feedback to improve the recommendations in the long run. Implement mechanisms to gather user reviews and ratings for destinations they have visited through your system. Utilize this feedback to refine your recommendation algorithms, enhance the accuracy of predictions, and provide even better suggestions to future users.") 
        map_data = {
                    # Define the map layout
                    'layout': go.Layout(
                    title='Places to visit by Location',
                    autosize=True,
                    hovermode='closest',
                    mapbox=dict(
                            style='stamen-terrain',
                            bearing=0,
                            center=dict(lat=8, lon=20),
                            pitch=0,
                            zoom=2
                        ),
                ),

                    # Define the map data as a scatter plot of the coordinates
                    'data': go.Scattermapbox(
                    lat=clean_df['latitude'],
                    lon=clean_df['longitude'],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=clean_df['rating'],
                        opacity=0.8
                        ),
                    text=['Price: ${}'.format(i) for i in clean_df['UpperPrice']],
                    hovertext=clean_df.apply(lambda x: f"Ranking Type: ${x['RankingType']}, Location: {x['locationString']}", axis=1),
                        )
                }

        # Create the map figure
        fig = go.Figure(data=[map_data['data']], layout=map_data['layout'])

        # Show the map figure in Streamlit
        st.plotly_chart(fig, width='100%')

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

        
#def _extracted_from_main_89():    
    if selection == "Recomenders" : 
        with st.sidebar.expander(""):
            pass    
        st.title("Recommender System")
    
        # Set the CSS style
        st.markdown("""
            <style>
                .st-sidebar {
                    width: 100px;
                }
            </style>
        """, unsafe_allow_html=True)

        # Get the user's preferences
        #preferences = st.multiselect("What are your preferences?", ["Hotel", "Restaurant", "Culture", "Specialty Lodging", "Bed and Breakfast","Pool", "Adventure"])


        # Recommend locations based on the user's preferences
        st.header("Attraction Recommendation")
        rating_threshold = st.number_input("Enter Rating Threshold", min_value=0.0, max_value=5.0, value=3.0, step=0.1)

        if st.button("Get Attraction Recommendations"):
            recommendations = recommender.recommend_attraction(rating_threshold, cosine_similarities, cosine_sim2)
            st.dataframe(recommendations)
         
        st.header("Place Recommendation")
        place_name = st.text_input("Enter Place Name")

        if st.button("Get Place Recommendations"):
            recommended_places = recommender.recommend_place(place_name, cosine_similarities, cosine_sim2)
            st.write(recommended_places)
            
                 
        st.header("Country Recommendation")
        Country_name = st.text_input("Enter Country Name")

        if st.button("Get Country Recommendations"):
            recommended_country = recommender.recommend_country(Country_name, cosine_similarities, cosine_sim2)
            st.write(recommended_country)
                     
        st.header("Subcatergory Recommendation")
        subcategories = st.text_input("Enter Subcatergory Name")

        if st.button("Get Subcatergory Recommendations"):
            recommended_subcategory = recommender.recommend_subcategory(subcategories, cosine_sim2, cosine_similarities)
            st.write(recommended_subcategory)

       
        amenities = clean_df['amenities'].str.split(', ').explode().unique()

        if len(amenities) >= 0:
            # Convert the amenities array to a list
            amenities_list = amenities.tolist()

            if selected_amenities := st.multiselect(
                "Select Amenities", amenities_list
            ):
                # Filter the clean_df DataFrame based on the selected amenities
                filtered_df = clean_df[clean_df['amenities'].str.contains('|'.join(selected_amenities))]

                # Get the recommendations for the selected amenities
                recommendations = recommender.select_amenities(selected_amenities[0], cosine_sim2, cosine_similarities)

                # Display the recommendations
                st.write(recommendations)
            else:
                st.write("Please select at least one amenity.")
        else:
            st.write("No amenities found.")
    if selection == "Gallery" :
        with st.sidebar.expander(""):
            pass
        
        def gallery(gallery_images):
            """Displays a gallery of images."""
            if len(gallery_images) > 0:
                st.markdown("### Gallery Images")
                num_images_per_row = 3
                columns = st.columns(num_images_per_row)
              
                for i, image in enumerate(gallery_images):
                    row_columns = columns[i % num_images_per_row]

                    image_name = image.split("/")[-1].split(".")[0]
                    row_columns.image(image, caption=image_name, width=400)
            else:
                st.markdown("No images found in the gallery.")

        gallery_images = glob.glob("../Data/images/gallery/*.jpeg")
        gallery(gallery_images)

        with st.form(key='gallery-form'):
            image_name = st.text_input("Image Name")
            st.markdown("### Message")
            message = st.text_area(label='Enter your message here')
            image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
            if st.form_submit_button("Add Image"):
                if image_file is not None:
                    image_path = f"../Data/images/gallery/{image_name}.jpeg"
                    with open(image_path, "wb") as f:
                        f.write(image_file.read())
                gallery_images.append(image_path)
                st.success("Image added successfully!")
            else:
                st.error("Please upload an image.")
                
if __name__ == "__main__":
    main()