# Africura Travel Destination Recommendation System
A recommendation system that gives users suggestions that best match their travel preferences in and around African countries.
<p>
    <img src="Data/images/readme_banner.jpg" alt="Banner Image"/>
</p>

#### Authors
* ![Dennis Mwanzia](https://github.com/DennisMwanzia)
* ![Pamela Owino](https://github.com/PamelaAwino)
* ![Joshua Rwanda](https://github.com/R3TR0Quan)
* ![Nelson Kemboi](https://github.com/nelkemboi)
* ![Pauline Wambui](https://github.com/paulineKiarie)
* ![Kane Muendo](https://github.com/kanevundi)
* ![Ian Macharia](https://github.com/Imacharia)

## Introduction

Tourists visiting Africa often struggle to find suitable travel destinations that align with their preferences, making it challenging to plan a satisfying trip within their budget and time constraints. 

Our main aim as AfricuraAI is to develop a machine learning model i.e. recommendation system that provides personalized recommendations for the best tourist destinations in Africa. By considering customer reviews, budget constraints, specific locations, available amenities, and residence type, the model aims to suggest the ideal tourist destination that aligns with the user's preferences.

## Objective

The goal is to build a machine learning model that can accurately predict hotel ratings based on customer reviews, budget constraints, specific locations, and the type of residence. The model will help users make informed decisions when selecting hotels by considering their preferences and constraints.

## Our Data

We sourced data by scraping destination review data from **TripAdvisor** 
Here's a breakdown of what each column represent based on the names:

* id: Unique identifier for each item.
* type: Type of the item.
* category: Category of the item.
* subcategories: Subcategories associated with the item.
* name: Name of the item.
* locationString: String representation of the location of the item.
* description: Description or details about the item.
* image: Image associated with the item.
* photoCount: Number of photos available for the item.
* awards: Awards received by the item.
* rankingPosition: Ranking position of the item.
* rating: Rating of the item.
* rawRanking: Raw ranking of the item.
* phone: Phone number associated with the item.
* address: Address of the item.
* addressObj: Address information in object format.
* localName: Local name of the item.
* localAddress: Local address of the item.
* localLangCode: Language code for the local information.
* email: Email address associated with the item.
* latitude: Latitude coordinate of the item's location.
* longitude: Longitude coordinate of the item's location.
* webUrl: URL associated with the item.
* website: Website URL of the item.
* rankingString: Ranking information in string format.
* rankingDenominator: Denominator for the ranking.
* neighborhoodLocations: Locations of the item in the neighborhood.
* nearestMetroStations: Nearest metro stations to the item.
* ancestorLocations: Ancestor locations of the item.
* ratingHistogram: Histogram data for the item's ratings.
* numberOfReviews: Number of reviews for the item.
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
* roomTips: Tips or recommendations for rooms (for hotels).
* checkInDate: Date for check-in (for hotels).
* checkOutDate: Date for check-out (for hotels).
* offers: Offers associated with the item.
* guideFeaturedInCopy: Information about guides featuring the item.
* isClosed: Indicates if the item is closed.
* isLongClosed: Indicates if the item has been closed for a long time.
* openNowText: Text indicating if the item is currently open.
* cuisines: Cuisines offered (for restaurants).
* mealTypes: Types of meals available (for restaurants).
* dishes: Dishes served (for restaurants).
* features: Features or highlights of the item.
* dietaryRestrictions: Dietary restrictions or considerations.
* hours: Operating hours of the item.
* menuWebUrl: URL for the menu (for restaurants).
* establishmentTypes: Types of establishments.
* ownersTopReasons: Top reasons provided by owners.
* rentalDescriptions: Descriptions related to rentals.
* photos: Photos associated with the item.
* bedroomInfo: Information about bedrooms (for accommodations).
* bathroomInfo: Information about bathrooms (for accommodations).
* bathCount: Number of bathrooms (for accommodations).
* baseDailyRate: Base daily rate (for accommodations).

One **challenge** we faced was that the data was extremely inconsistent and had to be *'wrangled'* to be able to use if for the model