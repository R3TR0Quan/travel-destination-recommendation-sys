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
            background-image: url('https://github.com/R3TR0Quan/travel-destination-recommendation-sys/tree/main/Data/images/ui_bg.jpg'); /* Enter the path to your background image here */
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
    st.markdown("<body><section id='search-section'><h2>Search for Your Ideal Destination</h2><form action='#' method='GET'><input type='text' name='price' placeholder='Price Range'><br><input type='text' name='place' placeholder='Location'><br><input type='text' name='amenities' placeholder='Amenities'><br><input type='submit' value='Search'></form></section></body>", unsafe_allow_html=True)
    # Add other sections using st.markdown()
    st.markdown("## About")
    st.markdown("Provide an overview of the Africura Travel Destination Recommendation System and its purpose. Describe how the system works and what it offers to users.")
    st.markdown("## Destinations")
    st.markdown("Present information about various travel destinations in Africa. Include descriptions, images, and highlights of each destination. You can provide links or buttons for users to explore more details about each destination.")
    st.markdown("## Contact")
    st.markdown("Include contact information or a contact form for users to get in touch with the Africura Travel Destination Recommendation System team. You can provide an email address, phone number, or social media handles.")
    st.markdown("<footer><p>&copy; 2023 Africura Travel Destination Recommendation System. All rights reserved.</p></footer>", unsafe_allow_html=True)
if __name__ == '__main__':
    main()