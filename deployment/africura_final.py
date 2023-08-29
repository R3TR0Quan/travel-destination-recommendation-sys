import pandas as pd
import numpy as np
import json
import glob
import re
import os
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
    with open(r'Data/clean_df.pkl', 'rb') as f:
        clean_df = pickle.load(f)

    # Load the pickled files
    with open(r'Data/.tfidf_matrix2.pkl', 'rb') as f:
        tfidfv_matrix2 = pickle.load(f)

    with open(r'Data/.cosine_sim2.pkl', 'rb') as f:
        cosine_sim2 = pickle.load(f)

    with open(r'Data/.cosine_similarities.pkl', 'rb') as f:
        cosine_similarities = pickle.load(f)

    with open(r'Data/.indices.pkl', 'rb') as f:
        indices = pickle.load(f)

    return clean_df, tfidfv_matrix2, cosine_sim2, cosine_similarities, indices

st.set_page_config(layout="wide")



    # Add background image and custom styles
page_bg_img_about = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://cdn.pixabay.com/photo/2019/09/20/23/47/sand-4492751_640.jpg");
    background-size: cover;
    background-position: top left;
    background-repeat: no-repeat;
}

[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}

[data-testid="Toolbar"] {
    right: 2rem;
}
</style>
"""
st.markdown(page_bg_img_about, unsafe_allow_html=True)


page_bg_img_recomender = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://cdn.pixabay.com/photo/2016/10/20/22/18/solitude-1756736_640.jpg");
    background-size: cover;
    background-position: top left;
    background-repeat: no-repeat;
}

[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}

[data-testid="Toolbar"] {
    right: 2rem;
}
</style>
"""
st.markdown(page_bg_img_recomender, unsafe_allow_html=True)


page_bg_img_gallery = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://cdn.pixabay.com/photo/2018/10/14/20/38/sunset-3747442_640.jpg");
    background-size: cover;
    background-position: top left;
    background-repeat: no-repeat;
}

[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}

[data-testid="Toolbar"] {
    right: 2rem;
}
</style>
"""
st.markdown(page_bg_img_gallery, unsafe_allow_html=True)



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
        ].astype({'LowerPrice': int, 'UpperPrice': int})

    def select_amenities(self, amenities, cosine_similarities, cosine_sim2):
        indices = {title: index for index, title in enumerate(self.clean_df['amenities'])}

        # Check if the combined_amenities exists in the indices dictionary
        if amenities in indices:
            idx = indices[amenities]
            sim_scores = list(enumerate(np.dot(self.cosine_sim2[idx], self.cosine_similarities)))
            sim_scores.sort(key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:11]
            indices = [x for x, _ in sim_scores]
            return self.clean_df.set_index('amenities').iloc[indices][['name', 'country', 'RankingType', 'subcategories', 'LowerPrice', 'UpperPrice']]
        else:
            return f"No recommendations found for {amenities}."


    def recommend_amenities(combined_amenities, cosine_sim2, cosine_similarity, clean_df):
        indices = {title: index for index, title in enumerate(clean_df['combined_amenities'])}
        idx = indices[combined_amenities]
        sim_scores = list(enumerate(np.dot(cosine_sim2[idx], cosine_similarities)))
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        indices = [x for x, _ in sim_scores]
        return clean_df.set_index('combined_amenities').iloc[indices][
            [
                'name',
                'country',
                'RankingType',
                'subcategories',
                'LowerPrice',
                'UpperPrice',
            ]
        ].astype({'LowerPrice': int, 'UpperPrice': int})
        
    
    def recommend_subcategory(self, subcategories, cosine_similarities, cosine_sim2):
        indices = {title: index for index, title in enumerate(self.clean_df['subcategories'])}
        idx = indices[subcategories]
        sim_scores = list(enumerate(np.dot(self.cosine_sim2[idx], self.cosine_similarities)))
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        indices = [x for x, _ in sim_scores]
        return self.clean_df.set_index('subcategories').iloc[indices][
            ['name', 'country', 'RankingType', 'type', 'LowerPrice', 'UpperPrice']
        ].astype({'LowerPrice': int, 'UpperPrice': int})    

    def recommend_attraction(self, rating_threshold, cosine_similarities, cosine_sim2):
        recommendations = self.clean_df[self.clean_df['rating'] < rating_threshold][
            ['name', 'LowerPrice', 'UpperPrice', 'amenities', 'type', 'country']
        ].astype({'LowerPrice': int, 'UpperPrice': int})
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
        ].astype({'LowerPrice': int, 'UpperPrice': int})

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
    



   # Add other sections using st.markdown()
    st.sidebar.header("Africura!")
    
                
    st.sidebar.subheader("About")
    st.sidebar.write("Africura is a recommendation engine that provides suggestions for locations to visit in Africa based on given preferences")

                
    if selection == "Join":
        # Add header
        st.markdown("<h1 style='color: rgba(3, 3, 3, 6)'>Africura Members</h1>", unsafe_allow_html=True)
        st.write("<h6 style='color: rgba(3, 3, 3, 6)'> Coming Soon! Watch this space.</h6>", unsafe_allow_html=True)
        create_account()
          
    
   
    if selection == "About" :
        with st.sidebar.expander(""):
            pass
        
        # Add header
        st.markdown("<h1 style='color: rgba(3, 3, 3, 6)'>Welcome to Africura</h1>", unsafe_allow_html=True)
        
        st.markdown("##### A one-stop shop for all you WanderLusters. ")
        
        
        
        st.write("<h6 style='color: Hex(#FFFFFF)'>This recommendation system uses machine learning to recommend places to visit in Africa. The system takes into account a user's interests, and budget to generate recommendations.</h1>", unsafe_allow_html=True)
     
        
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
                    hovertext=clean_df.apply(lambda x: f"Ranking Type: ${x['RankingType']}, Location: {x['Location']}", axis=1),
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
        # Add header
        st.markdown("<h1 style='color: rgba(3, 3, 3, 6)'>Africura Recomender</h1>", unsafe_allow_html=True)
        

        # Recommend locations based on the user's preferences
        st.header("Attraction Recommendation")
        st.write("Disclaimer: Prices are only an estimate")
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
        # Add header
        st.markdown("<h1 style='color: rgba(3, 3, 3, 6)'>Africura Gallery</h1>", unsafe_allow_html=True)
        # List of image URLs from Pixabay
        image_urls = [
                        "https://cdn.pixabay.com/photo/2018/03/17/10/05/wildlife-3233525_640.jpg",
                        "https://cdn.pixabay.com/photo/2018/01/21/18/54/water-buffalo-3097317_640.jpg",
                        "https://cdn.pixabay.com/photo/2018/03/03/10/02/architecture-3195322_640.jpg",
                        "https://cdn.pixabay.com/photo/2017/07/14/04/50/cheetah-2502782_640.jpg",
                        "https://cdn.pixabay.com/photo/2018/09/17/14/36/lion-3683994_640.jpg",
                        "https://cdn.pixabay.com/photo/2014/06/20/18/34/ostriches-373339_640.jpg",
                        "https://cdn.pixabay.com/photo/2016/02/24/06/35/cape-of-good-hope-1219192_640.jpg",
                        "https://cdn.pixabay.com/photo/2017/05/10/23/10/morocco-2302244_640.jpg",
                        # Add more image URLs from Pixabay
                        ]

        # Display the images in the gallery
        for image_url in image_urls:
            image_name = os.path.basename(image_url)  # Extract the name from the URL
            st.image(image_url, caption=image_name, use_column_width=True)
            
            
            
            
            
        def gallery(gallery_files):
            """Displays a gallery of images and videos."""
            if len(gallery_files) > 0:
                st.markdown("### Gallery")
                num_files_per_row = 3
                columns = st.columns(num_files_per_row)

                for i, file_path in enumerate(gallery_files):
                    row_columns = columns[i % num_files_per_row]
                    file_extension = file_path.split(".")[-1]

                    if file_extension in ["png", "jpg", "jpeg"]:
                        file_type = "image"
                        file_caption = file_path.split("/")[-1].split(".")[0]
                        row_columns.image(file_path, caption=file_caption, width=400)
                    elif file_extension == "mp4":
                        file_type = "video"
                        file_caption = file_path.split("/")[-1].split(".")[0]
                        row_columns.markdown(f"**{file_caption}**")
                        row_columns.video(file_path)
                    else:
                        continue

            else:
                st.markdown("No files found in the gallery.")

        gallery_files = glob.glob("Data/images/gallery/*.*")  
        gallery(gallery_files)

        with st.form(key='gallery-form'):
            file_name = st.text_input("File Name")
            st.markdown("### Message")
            message = st.text_area(label='Enter your message here')
            file = st.file_uploader("Upload File", type=["png", "jpg", "jpeg", "mp4"])

            if st.form_submit_button("Add File"):
                if file is not None:
                    file_extension = file.name.split(".")[-1]
                    if file_extension == "mp4":
                        file_type = "video"
                    else:
                        file_type = "image"

                file_path = f"Data/images/gallery/{file_name}.{file_extension}"
                with open(file_path, "wb") as f:
                    f.write(file.read())

                gallery_files.append(file_path)
                st.success(f"{file_type.capitalize()} added successfully!")
            else:
                st.error("Please upload a file.")
                
if __name__ == "__main__":
    main()