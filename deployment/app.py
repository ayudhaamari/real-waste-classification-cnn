# app.py

import streamlit as st
import eda
import prediction

# Set the page configuration for the Streamlit app
st.set_page_config(
    page_title="Waste Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Create a sidebar for navigation
    st.sidebar.title("♻️ Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Home", "📊 EDA", "🔍 Prediction"])

    if page == "🏠 Home":
        # Add sidebar content for the Home page
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 About the Model")
        accuracy = 0.82
        st.sidebar.write("🎯 Model Accuracy:")
        st.sidebar.progress(accuracy)
        st.sidebar.write(f"{accuracy:.2%}")
        st.sidebar.write("**🤔 What is Accuracy?**")
        st.sidebar.write("Accuracy measures how well our model correctly classifies waste items.")
        st.sidebar.write("**💡 What does this mean?**")
        st.sidebar.write(f"Our model correctly classifies {accuracy:.2%} of waste items, helping improve recycling efficiency.")

        st.sidebar.markdown("---")
        st.sidebar.subheader("♻️ Fun Facts")
        # Added more fun facts
        fun_facts = [
            "Proper waste classification can increase recycling rates by up to 50%!",
            "Recycling one aluminum can saves enough energy to run a TV for three hours.",
            "It takes 450 years for a plastic bottle to decompose in a landfill.",
            "Glass can be recycled endlessly without losing quality or purity.",
            "Recycling paper saves 17 trees and 7,000 gallons of water per ton of paper."
        ]
        st.sidebar.info(fun_facts[st.session_state.get('fun_fact_index', 0)])
        
        # Button to cycle through fun facts
        if st.sidebar.button("Next Fun Fact"):
            st.session_state['fun_fact_index'] = (st.session_state.get('fun_fact_index', 0) + 1) % len(fun_facts)
            st.rerun()

        # Main content for the Home page
        st.title("♻️ Welcome to Waste Classification Tool")
        st.write("""
        This application provides functionalities for Exploratory Data Analysis and 
        Prediction of waste types. Use the navigation pane on the left to 
        select the module you wish to utilize.
        """)
        
        # Display an image in the center column
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image("https://assets-a1.kompasiana.com/items/album/2021/03/14/dr-stone-fandomcom-1536x864-604dff978ede483a3b589c96.png?t=o&v=780",  
                     caption="Notes: Recycling is crucial for our environment", use_column_width=True)
        
        st.markdown("---")
        
        # Information about the dataset
        st.write("#### 📊 Dataset")
        st.info("""
        The dataset used is the RealWaste dataset, containing images of waste items across 9 major material types, 
        collected within an authentic landfill environment. This dataset provides a realistic representation of 
        waste items, allowing our model to learn from real-world examples.
        """)
        
        # Problem statement
        st.write("#### ⚠️ Problem Statement")
        st.warning("""
        Manual waste sorting is inefficient, leading to low recycling rates and increased environmental harm. 
        Our goal is to develop a deep learning-based waste classification system using a Convolutional Neural Network (CNN) 
        that can accurately classify at least 70% of waste images across 9 material categories. This automated 
        system aims to significantly improve recycling efficiency and reduce the environmental impact of improper 
        waste disposal.
        """)
        
        # Project objective
        st.write("#### 🎯 Objective")
        st.success("""
        This project aims to develop a Convolutional Neural Network (CNN) model capable of accurately classifying waste images into nine distinct material types:
        1. Cardboard - e.g., boxes, packaging
        2. Glass - e.g., bottles, jars
        3. Metal - e.g., cans, foil
        4. Paper - e.g., newspapers, magazines
        5. Plastic - e.g., bottles, containers
        6. Miscellaneous Trash - e.g., non-recyclable items
        7. Food Organics - e.g., fruit peels, vegetable scraps
        8. Textile Trash - e.g., old clothes, fabrics
        9. Vegetation - e.g., leaves, branches

        By leveraging deep learning techniques, we seek to automate and improve waste management efficiency, ultimately contributing to environmental sustainability. Our model will analyze visual characteristics such as shape, color, and texture to categorize waste items, helping to streamline recycling processes and reduce the environmental impact of improper waste disposal.

        The successful implementation of this project could lead to:
        - Increased recycling rates
        - Reduced contamination in recycling streams
        - Lower operational costs for waste management facilities
        - Enhanced public awareness about proper waste sorting
        """)

    elif page == "📊 EDA":
        # Run the Exploratory Data Analysis module
        eda.run()
    
    elif page == "🔍 Prediction":
        # Run the Prediction module
        prediction.run()

if __name__ == "__main__":
    main()