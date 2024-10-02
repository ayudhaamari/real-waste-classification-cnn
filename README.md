# 🗑️ Real Waste Classification CNN

![Waste Classification](https://assets-a1.kompasiana.com/items/album/2021/03/14/dr-stone-fandomcom-1536x864-604dff978ede483a3b589c96.png?t=o&v=780)

<table style="width: 100%; text-align: center; border-collapse: collapse;">
    <tr>
        <th style="padding: 10px;">Dataset</th>
        <th style="padding: 10px;">Deployment</th>
        <th style="padding: 10px;">Model</th>
    </tr>
    <tr>
        <td style="padding: 10px;">
            <a href="https://archive.ics.uci.edu/dataset/908/realwaste">
                <img src="https://img.shields.io/badge/Dataset-RealWaste-orange" alt="Dataset">
            </a>
        </td>
        <td style="padding: 10px;">
            <a href="https://huggingface.co/spaces/amariayudha/RealWaste_Prediction_Deep_Learning">
                <img src="https://img.shields.io/badge/Demo-Hugging%20Face-blue" alt="Hugging Face Demo">
            </a>
        </td>
        <td style="padding: 10px;">
            <a href="https://drive.google.com/drive/folders/174TT5ANFTS3_uztj8z8GCFGldxJErAId?usp=sharing">
                <img src="https://img.shields.io/badge/Model-Google%20Drive-green" alt="Model">
            </a>
        </td>
    </tr>
</table>

## 🌟 Project Overview

This project develops a Convolutional Neural Network (CNN) model for classifying waste images into nine distinct material types. Our goal is to automate and improve waste management efficiency, ultimately contributing to environmental sustainability.

### 🎯 Objective

Develop a deep learning-based waste classification system using a CNN that can accurately classify at least 80% of waste images across 9 material categories.

### 📊 Dataset

We use the [RealWaste dataset](https://archive.ics.uci.edu/dataset/908/realwaste), containing images of waste items across 9 major material types, collected within an authentic landfill environment.

## 🗑️ Waste Categories

1. 📦 Cardboard
2. 🍾 Glass
3. 🥫 Metal
4. 📰 Paper
5. 🥤 Plastic
6. 🚮 Miscellaneous Trash
7. 🍎 Food Organics
8. 👕 Textile Trash
9. 🌿 Vegetation

## 🚀 Features

- 📊 **Exploratory Data Analysis (EDA)**: Visualizes dataset distribution and sample images.
- 🔮 **Prediction**: Classifies waste images using the trained CNN model.
- 🖥️ **Interactive Web Interface**: Built with Streamlit for easy use and deployment.
- 📈 **Performance Metrics**: Provides detailed model evaluation including accuracy, precision, recall, and F1-score.
- 🔄 **Data Augmentation**: Employs techniques to increase dataset diversity and model robustness.

## 🛠 Technologies Used

- ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=Python&logoColor=white) Python
- ![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?style=flat-square&logo=TensorFlow&logoColor=white) TensorFlow
- ![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=Streamlit&logoColor=white) Streamlit
- ![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=Pandas&logoColor=white) Pandas
- ![Plotly](https://img.shields.io/badge/-Plotly-3F4F75?style=flat-square&logo=Plotly&logoColor=white) Plotly
- ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat-square&logo=Python&logoColor=white) Matplotlib
- ![Seaborn](https://img.shields.io/badge/-Seaborn-3776AB?style=flat-square&logo=Python&logoColor=white) Seaborn
- ![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?style=flat-square&logo=OpenCV&logoColor=white) OpenCV

## 🖥️ Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/real-waste-classification-cnn.git
   ```

2. Install required packages:
   ```bash
   pip install -r deployment/requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run deployment/app.py
   ```

### 🧠 The Brain Behind the Magic: InceptionV3

This project doesn't just use any CNN - it leverages the InceptionV3 architecture! Here's some details:

- 🏗️ **Architecture**: 48 layers deep, including convolutions, max pooling, average pooling, concatenations, dropouts, and fully connected layers.
- 🔍 **Inception Modules**: Utilizes parallel convolutions of different sizes for multi-scale processing.
- 🏋️ **Efficient Computing**: Employs factorized convolutions and aggressive dimension reductions.
- 🌐 **Global Context**: Incorporates auxiliary classifiers for better gradient flow and regularization.
- 📊 **Parameters**: Approximately 23.9 million parameters - powerful yet manageable!


## 📊 Model Performance

The current model achieves an accuracy of 82% on the test set. Detailed metrics:

- Precision: 0.83
- Recall: 0.81
- F1-Score: 0.82

Here's a visual representation of our model's learning curve:

![Model Learning Curve](images/performance.jpg)
*Learning curve showing the model's training and validation performance over epochs*

This plot provides insights into how the model's performance improved during the training process, helping to understand its learning dynamics and potential areas for optimization.

## 🌐 Deployment

The project is deployed on Hugging Face Spaces. You can access it [here](https://huggingface.co/spaces/amariayudha/RealWaste_Prediction_Deep_Learning).

## 🎥 Live Demo

![Waste Classification Demo](path/to/demo.gif)

Check out our waste classification model in action! This demo shows how easy it is to upload an image and get instant classification results.

## 🔮 Future Improvements

The project team is constantly working to enhance the waste classification system. Some areas being explored include:

- Implementing object detection to classify multiple waste items in a single image
- Expanding the dataset to include more diverse waste types
- Experimenting with newer architectures like EfficientNet for potentially improved performance

## 💡 Real-World Applications

This waste classification system has numerous potential applications:

1. **Smart Bins**: Automating waste sorting in public spaces
2. **Recycling Plants**: Enhancing sorting efficiency in recycling facilities
3. **Educational Tools**: Teaching proper waste segregation in schools and communities
4. **Waste Management Apps**: Integrating with mobile apps to help users properly dispose of items

## 🌍 Environmental Impact

By improving waste classification accuracy, this project aims to:

- Increase recycling rates
- Reduce contamination in recycling streams
- Lower the amount of waste sent to landfills
- Promote circular economy principles

## 📁 Project Structure

```
real-waste-classification-cnn/
│
├── deployment/
│   ├── visualization/       # Visualization scripts and assets
│   ├── app.py               # Main Streamlit application
│   ├── eda.py               # Exploratory Data Analysis script
│   ├── prediction.py        # Prediction functionality
│   └── requirements.txt     # Deployment requirements
│
├── realwaste-image-classification.ipynb  # Notebook for model training and analysis
├── realwaste-image-classification-inference.ipynb          # Notebook for model inference and testing
└── url.txt                  # Contains links to dataset, deployment, and model
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Ayudha Amari Hirtranusi** 
- 🌐 **Github**: [www.github.com/ayudhaamari](https://github.com/ayudhaamari)
- 💼 **LinkedIn**: [www.linkedin.com/in/ayudhaamari/](https://www.linkedin.com/in/ayudhaamari/)
- 📧 **Email**: amariayudha@gmail.com

## 🙏 Acknowledgements

- [RealWaste Dataset](https://archive.ics.uci.edu/dataset/908/realwaste) creators
- [Streamlit](https://streamlit.io/) for the amazing web app framework
- [TensorFlow](https://www.tensorflow.org/) team for the powerful deep learning library

---

⭐️ If you find this project useful, please consider giving it a star!