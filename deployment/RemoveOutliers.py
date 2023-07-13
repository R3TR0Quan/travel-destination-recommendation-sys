import streamlit as st
import folium
import pandas as pd
import numpy as np
from scipy import stats

# Load the data from the CSV file
clean_df = pd.read_csv('app_data/clean_data.csv')

# Detect outliers using z-score method
zscore_threshold = 3  # Adjust this threshold based on your data and requirements
outliers = clean_df[(clean_df['latitude'] >= -35) & (clean_df['latitude'] <= 37) & (clean_df['longitude'] >= -25) & (clean_df['longitude'] <= 60) &
                    (np.abs(stats.zscore(clean_df[['latitude', 'longitude']])) > zscore_threshold).any(axis=1)]

# Replace outliers with NaN values in the original DataFrame
clean_df.loc[outliers.index, ['latitude', 'longitude']] = None

# Create a map centered at a specific location
map_center = [8, 20]
m = folium.Map(location=map_center, zoom_start=2)

# Add markers to the map for each location
for _, row in clean_df.iterrows():
    lat, lon = row['latitude'], row['longitude']
    rating = row['rating']
    price = row['UpperPrice']
    ranking_type = row['RankingType']
    location_string = row['locationString']
    popup_text = f"Ranking Type: {ranking_type}, Location: {location_string}, Price: ${price}"

    if not pd.isnull(lat) and not pd.isnull(lon):
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color='blue',
            fill=True,
            fill_color='blue',
            popup=popup_text
        ).add_to(m)

# Display the map in Streamlit
st.markdown('<h1>Places to visit by Location</h1>', unsafe_allow_html=True)
st.markdown('<h3>Hover over the markers to see more details.</h3>', unsafe_allow_html=True)
st.write(m)
