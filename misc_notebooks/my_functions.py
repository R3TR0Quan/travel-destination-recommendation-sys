import pandas as pd
import numpy as np


import json
import glob
import re



class DataCleaning:
    def __init__(self):
        self.df = None

    def read_json_files(self, json_files):
    # Reads multiple JSON files and concatenates them into a single DataFrame
        dfs = []
        for file in json_files:
            with open(file, encoding='utf-8') as f:
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
