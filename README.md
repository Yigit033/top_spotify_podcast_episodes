# Top Spotify Podcast Episodes Analysis

This project analyzes the daily updated Spotify podcast dataset to predict podcast ratings using machine learning techniques, specifically employing the CatBoostRegressor model.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Explanation](#model-explanation)
- [Results](#results)
- [License](#license)

## Introduction

In this project, we aim to explore and analyze Spotify's top podcasts by utilizing various features from the dataset, such as duration, total episodes, and publishing information. The main objective is to predict the ratings of podcasts based on these features using regression techniques.

## Dataset

The dataset used in this project is sourced from Kaggle and contains various attributes related to Spotify's top podcasts. The original CSV file is stored inside a ZIP archive, which is extracted for use in the analysis.

### Key Features:
- `rank`: The rank of the podcast.
- `duration_ms`: The duration of the podcast in milliseconds.
- `show.total_episodes`: The total number of episodes for each podcast.
- `show.publisher`: The publisher of the podcast.
- `language`: The language of the podcast.
- `explicit`: Indicates if the podcast contains explicit content.
- `date`: The date when the data was collected.

## Installation

To run this project, you will need to have Python installed on your machine. Additionally, install the required packages using pip:

```bash
pip install numpy pandas matplotlib catboost scikit-learn feature-engine shap


## Usage    

To use the project, follow these steps:

Clone the repository:

git clone https://github.com/yourusername/top_spotify_podcast_episodes.git
cd top_spotify_podcast_episodes


Ensure you have the dataset (top-spotify-podcasts-daily-updated.zip) in the project directory.

Run the script:

python downloading_dataset.py


This script will perform the following operations:

Extract the dataset from the ZIP file.
Process the data, including handling duplicates and missing values.
Encode categorical features using Rare Label Encoding.
Split the data into training and testing sets.
Train a CatBoost regression model to predict podcast ratings.
Generate SHAP values to explain model predictions.
Model Explanation
The CatBoostRegressor model is used in this project, which is a gradient boosting algorithm known for its efficiency with categorical features. The model is trained on the processed dataset to predict the ratings of podcasts.

Evaluation
The model's performance is evaluated using the Root Mean Squared Error (RMSE) metric, which provides insight into how well the model is performing on both the training and testing datasets.

## Results
The results of the model training can be visualized using SHAP summary plots, which illustrate the impact of each feature on the model's predictions.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

