﻿# Parkinson-Prediction-Streamlit-App

## Overview
This project focuses on detecting Parkinson's Disease using machine learning models trained on vocal attributes. The dataset, sourced from Kaggle, contains various speech-related features that help differentiate between healthy individuals and those with Parkinson’s.

The machine learning pipeline involves data preprocessing, feature scaling, and model evaluation using metrics like accuracy and recall. Multiple models—including **Logistic Regression**, **Random Forest**, and **Support Vector Machine (SVM)**—were trained, and the best-performing model was selected based on evaluation metrics. The final model is saved using pickle and integrated into a Streamlit-based web application for real-time predictions.

### Streamlit Application
This web app enables users to detect Parkinson’s Disease by analyzing voice recordings. It extracts key vocal features using the `librosa` library and feeds them into the trained machine learning model to determine the likelihood of Parkinson’s.

### Key Features:
- **Voice Feature Extraction:** Extracts jitter, shimmer, harmonic-to-noise ratio, and other vocal parameters from uploaded `.wav` files.
- **Machine Learning Prediction:** Utilizes a pre-trained model to analyze extracted features and predict Parkinson’s Disease probability.
- **Interactive UI:** Allows users to upload audio files, view predictions, and explore feature visualizations through radar charts, bar graphs, and tables.
- **Health Insights:** Provides a risk assessment to guide users on potential next steps.

This project presents an innovative approach to early Parkinson’s detection, serving as a valuable tool for healthcare research and public awareness.


## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Parkinson-Prediction-Streamlit-App.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Parkinson-Prediction-Streamlit-App
    ```
3. Create a conda environment with Python 3.10:
    ```bash
    conda create --name <env-name> python=3.10
    ```
4. Activate the conda environment:
    ```bash
    conda activate <env-name>
    ```
5. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
2. Open your web browser and go to `http://localhost:8501` to access the app.

## Live Demo
Check out the live demo of the app deployed on Streamlit Community Cloud: [Parkinson's Disease Prediction App](https://pd-prediction.streamlit.app/)

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.



