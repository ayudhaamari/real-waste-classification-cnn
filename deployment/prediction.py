import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import tensorflow as tf
import time
import os

# Load your trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('transfer_learning_model.h5')

def preprocess_image(image):
    img = image.resize((299, 299))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

def predict(image):
    model = load_model()
    _, processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_names = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation']
    return {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}

def run():
    st.title('🔍 Waste Classification Prediction')

    # Example images
    example_images = ['cig_package.jpg', 'stella.jpg', 'water_bottle.jpg', 'textile_shoes.jpg', 'organic_eggs.jpg', 'men_metalic_pose.jpg', 'normal_men.jpg','uno.jpg']
    example_path = './visualization'  # Set the path 

    st.subheader("Choose an example image or upload your own:")

    # Initialize session state for the selected image
    if 'selected_image' not in st.session_state:
        st.session_state.selected_image = None

    # Create columns for example images
    cols = st.columns(4)
    for i, img_name in enumerate(example_images):
        with cols[i % 4]:
            img_path = os.path.join(example_path, img_name)
            
            # Display the preview image under the button
            st.image(img_path, width=100, caption=f'Example {i+1}')
            
            # Create the button for each example
            if st.button(f"Example {i+1}", key=f"example_{i}"):
                st.session_state.selected_image = img_path

    uploaded_file = st.file_uploader("Or upload your own image", type=["jpg", "jpeg", "png"])

    # Use session state to store the selected or uploaded image
    if uploaded_file is not None:
        st.session_state.selected_image = uploaded_file

    image = None
    if st.session_state.selected_image is not None:
        if isinstance(st.session_state.selected_image, str):  # Example image case
            image = Image.open(st.session_state.selected_image).convert('RGB')
        else:  # Uploaded image case
            image = Image.open(st.session_state.selected_image).convert('RGB')

    if image:
        # Create two columns for images
        col1, col2 = st.columns(2)

        # Display original image in the left column
        with col1:
            st.subheader("Selected Image")
            st.image(image, caption='Selected Image', use_column_width=True)

        # Add a button to start prediction
        if st.button("Start Prediction"):
            # Progress and status indicators
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Preprocess the image
            status_text.text('Preprocessing image...')
            resized_image, _ = preprocess_image(image)
            progress_bar.progress(33)
            time.sleep(0.5)  # Simulate processing time

            # Display resized image in the right column
            with col2:
                st.subheader("Resized Image (299x299 for Model)")
                st.image(resized_image, caption='Resized Image for Prediction (299x299)', use_column_width=True)

            # Make prediction
            status_text.text('Making prediction...')
            prediction = predict(image)
            progress_bar.progress(66)
            time.sleep(0.5)  # Simulate processing time

            # Analyze results
            status_text.text('Analyzing results...')
            predicted_class = max(prediction, key=prediction.get)
            confidence = prediction[predicted_class]
            progress_bar.progress(100)
            time.sleep(0.5)  # Simulate processing time

            # Clear the status text and progress bar
            status_text.empty()
            progress_bar.empty()

            # Display prediction results under the images
            st.subheader("Prediction Results")
            st.write(f"Predicted waste type: **{predicted_class}**")
            st.write(f"Confidence: {confidence:.2%}")

            # Display vertical bar chart of probabilities using Plotly
            fig = go.Figure(data=[go.Bar(
                x=list(prediction.keys()),
                y=list(prediction.values()),
                marker=dict(
                    color=list(prediction.values()),
                    colorscale='Viridis',
                    colorbar=dict(title='Probability')
                )
            )])
            fig.update_layout(
                title='Prediction Probabilities',
                xaxis_title='Waste Type',
                yaxis_title='Probability',
                height=500,
                width=700
            )
            st.plotly_chart(fig)

if __name__ == "__main__":
    run()
