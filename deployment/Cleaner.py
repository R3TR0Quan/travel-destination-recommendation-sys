# Importing necessary libraries
import pandas as pd
import json
import glob
import re
import string


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans


from surprise import Dataset, Reader, KNNBasic, SVD, NMF, KNNWithMeans, SVDpp
from surprise.model_selection import train_test_split
from surprise import accuracy as sup_accuracy
from surprise.prediction_algorithms.matrix_factorization import NMF
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.prediction_algorithms.matrix_factorization import SVDpp
from surprise.model_selection import cross_validate


import warnings
# Ignore future deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sns.set_style('darkgrid')


class DataCleaning:
    def __init__(self):
        self.df = None

    def read_json_files(self, json_files):
    # Reads multiple JSON files and concatenates them into a single DataFrame
        dfs = []
        for file in json_files:
            with open(file) as f:
                json_data = json.load(f)
                df = pd.DataFrame(json_data)
                dfs.append(df)
                self.df = pd.concat(dfs, ignore_index=True)
                
        return self.df
    
    def get_preview(self, df):
        # Returns a preview of the DataFrame
        return self.df.head()
    
    
    def get_info(self, df):
        # returns Info on the dataset
        return self.df.info()
    
    def get_shape(self, df):
        # returns shape of the dataset
        return self.df.shape


    def drop_columns(self, columns):
        # Drops specified columns from the DataFrame
        self.df.drop(columns=columns, inplace=True)

    def missing_values_percentage(self, df):
        # Calculates the percentage of missing values in each column
        column_percentages = self.df.isnull().sum() / len(self.df) * 100
        columns_with_missing_values = column_percentages[column_percentages > 0]
        return columns_with_missing_values.sort_values(ascending=False)

    def drop_above_threshold(self, threshold):
        # Drops columns with missing values percentage above the specified threshold
        column_percentages = self.missing_values_percentage(self.df)
        columns_with_missing_values = column_percentages[column_percentages > threshold]
        columns_to_drop = columns_with_missing_values.index.tolist()
        self.df.drop(columns=columns_to_drop, inplace=True)

    def split_price_range(self):
        # Splits the priceRange column into LowerPrice and UpperPrice columns
        self.df[['LowerPrice', 'UpperPrice']] = self.df['priceRange'].str.replace('KES', '').str.split(' - ', expand=True)
        self.df['LowerPrice'] = self.df['LowerPrice'].str.replace(',', '').astype(float)
        self.df['UpperPrice'] = self.df['UpperPrice'].str.replace(',', '').astype(float)

    def fill_missing_prices(self):
        # Fills missing values in LowerPrice and UpperPrice columns based on type (ATTRACTION or HOTEL)
        self.df.loc[self.df['type'] == 'ATTRACTION', 'LowerPrice'] = self.df.loc[self.df['type'] == 'ATTRACTION', 'LowerPrice'].fillna(self.df['LowerPrice'].min())
        self.df.loc[self.df['type'] == 'ATTRACTION', 'UpperPrice'] = self.df.loc[self.df['type'] == 'ATTRACTION', 'UpperPrice'].fillna(self.df['UpperPrice'].min())
        self.df.loc[self.df['type'] == 'HOTEL', 'LowerPrice'] = self.df.loc[self.df['type'] == 'HOTEL', 'LowerPrice'].fillna(self.df['LowerPrice'].mean())
        self.df.loc[self.df['type'] == 'HOTEL', 'UpperPrice'] = self.df.loc[self.df['type'] == 'HOTEL', 'UpperPrice'].fillna(self.df['UpperPrice'].mean())

    def clean_amenities(self):
        # Formats the amenities column to have consistent formatting
        self.df['amenities'] = self.df['amenities'].apply(lambda x: ', '.join(['{:.2f}'.format(i) if isinstance(i, (float, int)) else str(i) for i in x]) if isinstance(x, (list, tuple)) else x)

    def replace_nan_amenities(self):
        # Replaces NaN values in the amenities column based on type (RESTAURANT or ATTRACTION)
        self.df.loc[(self.df['type'] == 'RESTAURANT') & (self.df['amenities'].isna()), 'amenities'] = 'restaurant'
        self.df.loc[(self.df['type'] == 'ATTRACTION') & (self.df['amenities'].isna()), 'amenities'] = 'bathroom only'

    def populate_empty_lists(self, new_data):
        # Populates empty lists in the amenities column with new_data
        self.df['amenities'] = [new_data if isinstance(value, list) and not value else value for value in self.df['amenities']]

    def extract_ranking_info(self):
        # Extracts ranking information from the rankingString column and creates new columns
        self.df['RankingType'] = ""
        self.df['Location'] = ""
        self.df['Numerator'] = ""
        self.df['Denominator'] = ""

        for index, row in self.df.iterrows():
            if pd.isnull(row['rankingString']):
                continue

            if match := re.match(r'#(\d+)\s+of\s+(\d+)\s+(.*?)\s+in\s+(.*?)$', row['rankingString']):
                numerator = match[1]
                denominator = match[2]
                ranking_type = match[3]
                location = match[4]

                self.df.at[index, 'RankingType'] = ranking_type
                self.df.at[index, 'Location'] = location
                self.df.at[index, 'Numerator'] = numerator
                self.df.at[index, 'Denominator'] = denominator

    def replace_ranking_types(self, mappings):
        # Replaces ranking types in the RankingType column based on specified mappings
        self.df['RankingType'] = self.df['RankingType'].replace(mappings)

    def split_ranking_string(self):
        # Splits the rankingString column into separate columns (Rank, Total, Location)
        self.df[['Rank', 'Total', 'Location']] = self.df['rankingString'].str.split(' of ', expand=True)
        self.df[['Total', 'rankingtype']] = self.df['Total'].str.split(' ', n=1, expand=True)
        self.df['Rank'] = self.df['Rank'].str.replace('#', '')
        self.df['Total'] = self.df['Total'].str.replace('things to do', '').str.replace('hotels', '').str.strip()
        self.df['Total'] = self.df['Total'].str.replace(",", "")
        self.df['Total'] = pd.to_numeric(self.df['Total'], errors='coerce')
        self.df['Rank'] = pd.to_numeric(self.df['Rank'], errors='coerce')

    def calculate_regional_rating(self):
        # Calculates the regional rating by dividing Total by Rank
        self.df['regional_rating'] = (self.df['Total'] / self.df['Rank']).astype(float)

    def fill_ranking_type(self, type_mapping):
        # Fills missing values in the RankingType column based on type_mapping and sets default values for VACATION_RENTAL and RESTAURANT types
        self.df['RankingType'] = np.where((self.df['RankingType'] == '') & (self.df['type'].map(type_mapping) != ''), self.df['type'].map(type_mapping), self.df['RankingType'])
        self.df['RankingType'] = self.df['RankingType'].fillna('VACATION_RENTAL').replace('VACATION_RENTAL', 'Specialty lodging')
        self.df['RankingType'] = self.df['RankingType'].fillna('RESTAURANT').replace('RESTAURANT', 'places to eat')

    def clean_ratings(self):
        # Replaces missing values in the rating column with 0
        self.df['rating'].fillna(0, inplace=True)

    def clean_review_tags(self):
        # Cleans up the reviewTags column by extracting the text values
        self.df.loc[:, 'reviewTags'] = self.df['reviewTags'].apply(lambda entries: [{'text': entry['text']} for entry in entries] if isinstance(entries, list) else [])
        self.df.loc[:, 'reviewTags'] = self.df['reviewTags'].apply(lambda tags: [tag['text'] for tag in tags])

    def fill_missing_coordinates(self):
        # Fills missing values in the longitude and latitude columns using interpolation
        self.df['longitude'] = self.df['longitude'].interpolate()
        self.df['latitude'] = self.df['latitude'].interpolate()
        self.df['longitude'] = self.df['longitude'].fillna(method='bfill')
        self.df['latitude'] = self.df['latitude'].fillna(method='bfill')

    def remove_outliers(self, outlier_latitudes, outlier_longitudes):
        # Removes rows with outlier coordinates
        precision = 2
        self.df['latitude'] = self.df['latitude'].round(precision)
        self.df['longitude'] = self.df['longitude'].round(precision)
        self.df = self.df.loc[~((self.df['latitude'].isin(outlier_latitudes)) & (self.df['longitude'].isin(outlier_longitudes))), :]

    def clean_subcategories(self):
        # Formats the subcategories column to have consistent formatting
        self.df['subcategories'] = self.df['subcategories'].apply(lambda x: ', '.join(['{:.2f}'.format(i) if isinstance(i, (float, int)) else str(i) for i in x]) if isinstance(x, (list, tuple)) else x)
        self.df.loc[self.df['type'] == 'VACATION_RENTAL', 'subcategories'] = self.df.loc[self.df['type'] == 'VACATION_RENTAL', 'subcategories'].fillna('Specialty Lodging')

    def drop_missing_values(self, columns):
        # Drops rows with missing values in specified columns
        self.df = self.df.dropna(subset=columns)

    def extract_country_and_city(self):
        # Extracts country and city information from the addressObj column
        self.df['country'] = self.df['addressObj'].apply(lambda x: x['country'] if isinstance(x, dict) else None)
        self.df['city'] = self.df['addressObj'].apply(lambda x: x['city'] if isinstance(x, dict) else None)

    def drop_unused_columns(self, columns):
        # Drops unused columns from the DataFrame
        self.df = self.df.drop(columns=columns)

    def replace_empty_strings(self):
        # Replaces empty strings with NaN values
        self.df = self.df.replace('', np.nan)

    def drop_rows_with_nan(self):
        # Drops rows with NaN values
        self.df = self.df.dropna()

    def save_to_csv(self, file_path):
        # Saves the DataFrame to a CSV file
        self.df.to_csv(file_path, index=False)



class DataProcessor:
    def __init__(self, clean_df):
        self.clean_df = clean_df
        self.clean_df_scaled = None
        
    def combine_amenities(self, row):
        if row is None:
            return None
        
        combined_items = set()
        for items in row:
            if items is not None:
                combined_items.update(items.split(", "))
        
        if len(combined_items) < 10:
            return ', '.join(row)  # Return the original row as a string
        
        return ', '.join(list(combined_items)[:10])
    
    

    def encode_columns(self):
        def map_column(column_name):
            unique_values = list(self.clean_df[column_name].unique())
            mapping = {}
            for index, value in enumerate(unique_values):
                mapping[value] = index + 1
            self.clean_df[column_name + "_mapped"] = self.clean_df[column_name].map(mapping)
        
        map_column("subcategories")
        map_column("amenities")
        map_column("RankingType")
        map_column("country")
        map_column("type")
        
    def process_data(self):
        self.clean_df = self.clean_df
        self.clean_df['combined_amenities'] = self.clean_df['amenities'].str.split(', ').apply(self.combine_amenities)
        self.encode_columns()
        
        # Select the numerical columns for normalization
        numerical_columns = ['rating', 'Rank', 'Total', 'regional_rating', 'LowerPrice', 'UpperPrice']
        
        # Normalize the numerical columns using normalize()
        normalized_data = normalize(self.clean_df[numerical_columns])
        self.clean_df_norm = self.clean_df.copy()
        self.clean_df_norm[numerical_columns] = normalized_data
        
        # Apply MinMaxScaler to 'rating' and 'Rank' columns
        scaler = MinMaxScaler()
        self.clean_df_scaled = self.clean_df.copy()
        self.clean_df_scaled[['rating', 'regional_rating', 'Rank']] = scaler.fit_transform(self.clean_df[['rating', 'regional_rating', 'Rank']])
        
        return self.clean_df, self.clean_df_scaled, self.clean_df_norm, self.clean_df['combined_amenities']

class PerformanceMetrics:
    def __init__(self, threshold, predictions):
        self.threshold = threshold
        self.predictions = predictions
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.precision = 0.0
        self.recall = 0.0

    def calculate_metrics(self):
        for prediction in self.predictions:
            if prediction[3] >= self.threshold:
                if prediction[2] >= self.threshold:
                    self.true_positives += 1
                else:
                    self.false_positives += 1
            elif prediction[2] >= self.threshold:
                self.false_negatives += 1
        
        if (self.true_positives + self.false_positives) != 0:
            self.precision = self.true_positives / (self.true_positives + self.false_positives)

        if (self.true_positives + self.false_negatives) != 0:
            self.recall = self.true_positives / (self.true_positives + self.false_negatives)

    def display_metrics(self):
        print(f"Precision: {self.precision:.2f}")
        print(f"Recall: {self.recall:.2f}")

clean_df = pd.read_csv("Data\clean_data.csv")

# creating a relevant columns from the above dataset 
vectorization_columns = clean_df[['name', 'subcategories', 'amenities']]
# Convert relevant data into a list of strings
documents = []
for _, row in vectorization_columns.iterrows():
    name = row['name']
    subcategories = row['subcategories']
    amenities = row['amenities']
    doc = f"{name} {subcategories} {amenities}"
    documents.append(doc)

# Apply TF-IDF vectorization
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(documents)

# Compute cosine similarity matrix
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

# Construct the TF-IDF Matrix
tfidfv2=TfidfVectorizer(analyzer='word', stop_words='english')
tfidfv_matrix2=tfidfv2.fit_transform(clean_df['description'])
#print(tfidfv_matrix2.todense())
#tfidfv_matrix2.todense().shape

# Calculate similarity matrix
cosine_sim2 = cosine_similarity(tfidfv_matrix2, tfidfv_matrix2)

# Create a Pandas Series to map place titles to their indices
indices = pd.Series(data = list(clean_df.index), index = clean_df['name'])




# this is a function to recomend places
def recommend_place(name, cosine_similarity, cosine_sim2, clean_df):
    # Create a dictionary to map place name to their indices
    indices = {title: index for index, title in enumerate(clean_df['name'])}

    # Get the index of the place that matches the name
    idx = indices[name]

    # Get the pairwise similarity scores of all places with that name
    sim_scores = list(enumerate((np.dot(cosine_sim2[idx], cosine_similarities))))

    # Sort the places based on the similarity scores
    sim_scores.sort(key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar names
    sim_scores = sim_scores[1:11]

    # Get the names indices
    indices = [x for x, _ in sim_scores]

    # Return the top 10 most similar place names
    recommended_place = clean_df.set_index('name').iloc[indices][
            [
                'country',
                'RankingType',
                'subcategories',
                'LowerPrice',
                'UpperPrice',
            ]
        ]

    return recommended_place


# this is a functtion that recommends amenities
def recommend_amenities(combined_amenities, cosine_sim2, cosine_similarity, clean_df):
    # Create a dictionary to map  amenities to their indices
    indices = {title: index for index, title in enumerate(clean_df['combined_amenities'])}

    # Get the index of the amenity that matches
    idx = indices[combined_amenities]

    # Get the pairwise similarity scores of all amenities with that amenity
    sim_scores = list(enumerate(np.dot(cosine_sim2[idx], cosine_similarities)))

    # Sort the amenities based on the similarity scores
    sim_scores.sort(key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar amenities
    sim_scores = sim_scores[1:11]

    # Get the amenities indices
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
    ]
    

# this is a function that recomends attraction sites based on the rating

def recommend_attraction(cosine_sim2, cosine_similarities, rating_threshold):
    # Filter the DataFrame based on the rating threshold
    recommendations = clean_df[clean_df['rating'] == rating_threshold][['name', 'LowerPrice', 'UpperPrice','amenities', 'type', 'country']]
    # Reset the index of the recommendations DataFrame
    recommendations.reset_index(drop=True, inplace=True)

    return recommendations

# this is a function that recommends based on the country
def recommend_country(country, cosine_sim2, cosine_similarities, data):
    # Create a dictionary to map country titles to their indices
    indices = {title: index for index, title in enumerate(clean_df['country'])}

    # Get the index of the country
    # that matches the title
    idx = indices[country]

    # Get the pairwise similarity scores of all countries with that name
    sim_scores = list(enumerate(np.dot(cosine_sim2[idx], cosine_similarities)))

    # Sort the countries based on the similarity scores
    sim_scores.sort(key=lambda x: x[1], reverse=True)

    # Get the scores of the most similar countries
    sim_scores = sim_scores[1:50]

    # Get the country indices
    indices = [x for x, _ in sim_scores]

    # Return the top most similar countries
    recommended_country = clean_df.set_index('country').iloc[indices][
            [
                'name',
                'city',
                'RankingType',
                'subcategories',
                'LowerPrice',
                'UpperPrice',
            ]
        ]

    filtered_recommendations = recommended_country[recommended_country.index == country]

    return pd.DataFrame(filtered_recommendations)

    

class RecommendationEngine:
    def __init__(self, cosine_similarities, cosine_sim2, clean_df):
        self.cosine_similarities = cosine_similarities
        self.cosine_sim2 = cosine_sim2
        self.clean_df = clean_df

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

    def recommend_attraction(self, rating_threshold):
        recommendations = self.clean_df[self.clean_df['rating'] == rating_threshold][['name', 'LowerPrice', 'UpperPrice', 'amenities', 'type', 'country']]
        recommendations.reset_index(drop=True, inplace=True)
        return recommendations

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
