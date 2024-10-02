# Real Waste Classification CNN

![Waste Classification](https://assets-a1.kompasiana.com/items/album/2021/03/14/dr-stone-fandomcom-1536x864-604dff978ede483a3b589c96.png?t=o&v=780)

## 🌟 Project Overview

This project aims to develop a Convolutional Neural Network (CNN) model for classifying waste images into nine distinct material types. Our goal is to automate and improve waste management efficiency, ultimately contributing to environmental sustainability.

### 🎯 Objective

Develop a deep learning-based waste classification system using a CNN that can accurately classify at least 70% of waste images across 9 material categories.

### 📊 Dataset

We use the [RealWaste dataset](https://archive.ics.uci.edu/dataset/908/realwaste), containing images of waste items across 9 major material types, collected within an authentic landfill environment.

## 🗑️ Waste Categories

1. Cardboard
2. Glass
3. Metal
4. Paper
5. Plastic
6. Miscellaneous Trash
7. Food Organics
8. Textile Trash
9. Vegetation

## 🚀 Features

- **Exploratory Data Analysis (EDA)**: Visualize dataset distribution and sample images.
- **Prediction**: Classify waste images using the trained CNN model.
- **Interactive Web Interface**: Built with Streamlit for easy use and deployment.

## 🛠️ Technologies Used

- Python
- TensorFlow
- Streamlit
- Pandas
- Plotly
- Matplotlib
- Seaborn

## 🖥️ Installation & Usage

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/real-waste-classification-cnn.git
   ```

2. Install required packages:
   ```
   pip install -r deployment/requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run deployment/app.py
   ```

## 📊 Model Performance

Our current model achieves an accuracy of 82% on the test set.

## 🌐 Deployment

The project is deployed on Hugging Face Spaces. You can access it [here](https://huggingface.co/spaces/amariayudha/RealWaste_Prediction_Deep_Learning).

## 📁 Project Structure
