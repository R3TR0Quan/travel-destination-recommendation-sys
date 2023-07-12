import streamlit as st

def main():
    # Set the CSS style
    css = '''
    <style>
        /* Global styles */
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-image: url('travel-destination-recommendation-sys\Data\images\lina-loos-04-C1NZk1hE-unsplash.jpg'); /* Enter the path to your background image here */
            background-size: cover;
            background-position: center;
        }

        /* Header styles */
        header {
            background-color: white;
            color: #fff;
            padding: 20px;
            text-align: center;
        }

        h1 {
            margin: 0;
        }

        /* Navigation styles */
        nav {
            background-color: #f2f2f2;
            padding: 10px;
            text-align: center;
        }

        ul {
            list-style-type: none;
            margin: 0;
            padding: 0;
        }

        li {
            display: inline;
            margin-right: 10px;
        }

        a {
            color: #333;
            text-decoration: none;
            padding: 5px;
        }

        /* Section styles */
        section {
            padding: 20px;
            margin-bottom: 20px;
            background-color: #f9f9f9;
            opacity: 0.9;
            border-radius: 10px;
        }

        h2 {
            margin-top: 0;
        }

        /* Search section styles */
        #search-section {
            background-color: rgba(255, 255, 255, 0.9);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin: 20px auto;
            width: 80%;
            max-width: 600px;
        }

        input[type="text"] {
            padding: 10px;
            width: 100%;
            margin-bottom: 10px;
            border-radius: 5px;
            border: 1px solid #ccc;
        }

        input[type="submit"] {
            padding: 10px 20px;
            background-color: #333;
            color: #fff;
            border: none;
            cursor: pointer;
            border-radius: 5px;
        }

        /* Footer styles */
        footer {
            background-color: #333;
            color: #fff;
            padding: 10px;
            text-align: center;
        }
    </style>
    '''

    # Render the HTML code
    st.markdown(css, unsafe_allow_html=True)
    st.markdown("<header><h1>Africura Travel Destination Recommendation System</h1></header>", unsafe_allow_html=True)
    
    # Add search section
    st.markdown("<section id='search-section'><h2>Search for Your Ideal Destination</h2><form action='#' method='GET'><input type='text' name='price' placeholder='Price Range'><br><input type='text' name='place' placeholder='Location'><br><input type='text' name='amenities' placeholder='Amenities'><br><input type='submit' value='Search'></form></section>", unsafe_allow_html=True)

    # Add other sections using st.markdown()
    st.markdown("## About")
    st.markdown("Welcome to Africura Travel Destination Recommendation System, your gateway to unforgettable adventures across Africa. Discover breathtaking landscapes, vibrant cultures, and extraordinary wildlife as we guide you to the most awe-inspiring destinations on the continent. ")

    st.markdown("## Destinations")
    st.markdown("Discover the wonders of Kenya's Maasai Mara and the majestic Ethiopian rock-hewn churches through Africura Travel Destination Recommendation, your ultimate guide to unforgettable African adventures. Immerse yourself in the ancient treasures of Egypt, explore the vibrant medinas of Morocco, and experience the breathtaking wildlife in South Africa, Tanzania, Uganda, Malawi, Cape Verde, and more with the expertise of Africura's travel recommendations.")

    st.markdown("## Contact")
    st.markdown("Email: info@africuratravel.com, Telephone: +254 723 430 921, Facebook: facebook.com/africuratravel, Twitter: twitter.com/africuratravel.")

    st.markdown("<footer><p>&copy; 2023 Africura Travel Destination Recommendation System. All rights reserved.</p></footer>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
