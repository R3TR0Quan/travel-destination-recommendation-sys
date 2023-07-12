# Africura Travel Destination Recommendation System
A recommendation system that gives users suggestions that best match their travel preferences in and around African countries.
<p>
    <img src="Data/images/readme_banner.jpg" alt="Banner Image"/>
</p>
<p align="center">
    <img src="https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white&style=flat-square">
    <img src="https://img.shields.io/badge/-Surprise-4B0082?logo=python&logoColor=white&style=flat-square">
    <img src="https://img.shields.io/badge/-Streamlit-FF4B4B?logo=streamlit&logoColor=white&style=flat-square">
    <img src="https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white&style=flat-square">
    <img src="https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white&style=flat-square">
    <img src="https://img.shields.io/badge/-NLTK-4EA94B?logo=python&logoColor=white&style=flat-square">
</p>

#### Authors
* [Dennis Mwanzia](https://github.com/DennisMwanzia)
* [Pamela Owino](https://github.com/PamelaAwino)
* [Joshua Rwanda](https://github.com/R3TR0Quan)
* [Nelson Kemboi](https://github.com/nelkemboi)
* [Pauline Wambui](https://github.com/paulineKiarie)
* [Kane Muendo](https://github.com/kanevundi)
* [Ian Macharia](https://github.com/Imacharia)

## Introduction

Tourists visiting Africa often struggle to find suitable travel destinations that align with their preferences, making it challenging to plan a satisfying trip within their budget and time constraints. 

Our main aim as AfricuraAI is to develop a machine learning model i.e. recommendation system that provides personalized recommendations for the best tourist destinations in Africa. By considering customer reviews, budget constraints, specific locations, available amenities, and residence type, the model aims to suggest the ideal tourist destination that aligns with the user's preferences.

## Objective

The goal is to build a machine learning model that can accurately predict hotel ratings based on customer reviews, budget constraints, specific locations, and the type of residence. The model will help users make informed decisions when selecting hotels by considering their preferences and constraints.

## Our Data

We sourced data by scraping destination review data from **TripAdvisor** 
Here's a breakdown of aome of the main columns we used in coming up with recommendations:

* id: Unique identifier for each item.
* type: Type of the item.
* category: Category of the item.
* subcategories: Subcategories associated with the item.
* name: Name of the item.
* rankingPosition: Ranking position of the item.
* rating: Rating of the item.
* rawRanking: Raw ranking of the item.
* addressObj: Address information in object format.
* latitude: Latitude coordinate of the item's location.
* longitude: Longitude coordinate of the item's location.
* webUrl: URL associated with the item.
* website: Website URL of the item.
* rankingString: Ranking information in string format.
* rankingDenominator: Denominator for the ranking.
* neighborhoodLocations: Locations of the item in the neighborhood.
* ratingHistogram: Histogram data for the item's ratings.
* reviewTags: Tags associated with the reviews.
* reviews: Reviews of the item.
* booking: Booking information for the item.
* offerGroup: Group of offers associated with the item.
* subtype: Subtype or specific type of the item.
* hotelClass: Class or rating of a hotel item.
* hotelClassAttribution: Attribution information for the hotel's class.
* amenities: Amenities available at the item.
* numberOfRooms: Number of rooms available (for hotels).
* priceLevel: Price level or range of the item.
* priceRange: Price range of the item.

There were a lot more columns in the data we scraped such as `localname`, `cuisine` and `email` that were dropped.
We also conducted feature engineering on some columns to capture more information. All this is well documented in the included project **writeup**.

<p align='center'>
    <b>File Hierarchy
</p>
<p align='center'>
├── Data
│   ├── botswana.json
│   ├── Botswana reviews.json
│   ├── capeverde.json
│   ├── clean_data.csv
│   ├── compiled_data.csv
│   ├── condensed_data.csv
│   ├── drc.json
│   ├── DRC reviews.json
│   ├── egypt.json
│   ├── Egypt reviews.json
│   ├── ethiopia.json
│   ├── ghana.json
│   ├── images
│   │   ├── Columnstoplot.png
│   │   ├── map.png
│   │   ├── multicollinearity.png
│   │   ├── readme_banner.jpg
│   │   ├── top_10_subcategories_individually.png
│   │   ├── top_subcategories.png
│   │   └── ui_bg.jpg
│   ├── kenya.json
│   ├── madagascar.json
│   ├── malawi.json
│   ├── morocco.json
│   ├── namibia.json
│   ├── nigeria.json
│   ├── reviews_data.csv
│   ├── rwanda.json
│   ├── senegal.json
│   ├── seychelles.json
│   ├── south_africa.json
│   ├── southafrica.json
│   ├── tanzania.json
│   ├── Tanzania reviews.json
│   ├── uganda.json
│   └── zambia.json
├── deployment
│   └── africura.py
├── LICENSE
├── misc_notebooks
│   ├── dennis.ipynb
│   ├── ian.ipynb
│   ├── kane.ipynb
│   ├── kibet.ipynb
│   ├── pamela.ipynb
│   ├── Pauline.ipynb
│   └── rwanda.ipynb
├── notebook.ipynb
├── README.md
└── writeup.docx
</p>