# eda.py

# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Function to load data with caching for performance
@st.cache_data
def load_data():
    return pd.DataFrame({
        'Class': ['Plastic', 'Metal', 'Paper', 'Miscellaneous Trash', 'Cardboard', 'Vegetation', 'Glass', 'Food Organics', 'Textile Trash'],
        'Number of Images': [921, 790, 500, 495, 461, 436, 420, 411, 318]
    })

# Main function to run the Streamlit app
def run():
    st.title('📊 Exploratory Data Analysis - Waste Classification')

    # Load the data
    data = load_data()

    # Create a selectbox for users to choose visualization
    visualization_option = st.selectbox(
        "Choose a visualization:",
        ("Dataset Information and Distribution", "Sample Images")
    )

    if visualization_option == "Dataset Information and Distribution":
        st.subheader("Dataset Information and Distribution")
        
        # Add checkbox for showing dataset information
        show_dataset_info = st.checkbox("Show Dataset Information", value=True)
        
        if show_dataset_info:
            st.write(data)
            st.write("The dataset shows an uneven distribution across the nine waste categories. "
                     "This imbalance may impact model performance and will need to be addressed during the model training phase.")

        # Bar chart
        fig_bar = px.bar(data, x='Class', y='Number of Images', color='Class',
                     title='Number of Images per Waste Category')
        st.plotly_chart(fig_bar, use_container_width=True)

        # Pie chart
        fig_pie = px.pie(data, values='Number of Images', names='Class',
                         title='Proportion of Images per Waste Category')
        st.plotly_chart(fig_pie, use_container_width=True)

        st.write("These charts show the distribution of images across different waste categories. "
                 "Plastic and Metal categories have significantly more images, which could lead to bias in the model.")

    elif visualization_option == "Sample Images":
        st.subheader("Sample Images")
        st.write("Here are sample images from each waste category:")
        
        categories = ['cardboard', 'food_organics', 'glass', 'metal', 'misc', 'paper', 'plastic', 'textile', 'vegetation']
        
        # Create a selectbox for choosing a specific category
        selected_category = st.selectbox("Select a waste category:", categories)
        
        st.write(f"**{selected_category.capitalize()}**")
        
        cols = st.columns(3)
        for i in range(1, 4):
            with cols[i-1]:
                img_path = f'./visualization/{selected_category} ({i}).jpg'
                st.image(img_path, caption=f'{selected_category.capitalize()} ({i})', use_column_width=True)
        
        st.write("These sample images provide a visual representation of the selected waste category in our dataset.")
        
        # Add an option to view all categories
        if st.checkbox("View all categories"):
            for category in categories:
                if category != selected_category:
                    st.write(f"**{category.capitalize()}**")
                    cols = st.columns(3)
                    for i in range(1, 4):
                        with cols[i-1]:
                            img_path = f'./visualization/{category} ({i}).jpg'
                            st.image(img_path, caption=f'{category.capitalize()} ({i})', use_column_width=True)
                    st.markdown("---")  # Add a horizontal line after each category

# Entry point of the script
if __name__ == "__main__":
    run()