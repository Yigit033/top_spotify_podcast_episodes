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
