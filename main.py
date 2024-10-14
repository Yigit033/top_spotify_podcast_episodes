# Import necessary libraries
import numpy as np  # Import NumPy for handling numerical operations
import pandas as pd  # Import Pandas for data manipulation and analysis
import warnings  # Import Warnings to suppress unnecessary warnings

# Suppress warning messages
warnings.filterwarnings("ignore")

# Import SHAP for interpreting model predictions
import shap

# Import matplotlib for data visualization
import matplotlib.pyplot as plt

# Import CatBoostRegressor for building a regression model
from catboost import Pool, CatBoostRegressor

# Import mean_squared_error for evaluating model performance
from sklearn.metrics import mean_squared_error

# Import train_test_split for splitting the data into training and testing sets
from sklearn.model_selection import train_test_split

# Import RareLabelEncoder from feature_engine.encoding for encoding categorical features
from feature_engine.encoding import RareLabelEncoder

# Import CountVectorizer from sklearn.feature_extraction.text for text feature extraction
from sklearn.feature_extraction.text import CountVectorizer

# Import ast and re for working with text and regular expressions
import ast
import re

# Set Pandas options to display a maximum of 1000 rows
pd.set_option('display.max_rows', 1000)



##Normally we can read the csv file directly, but in this case, the csv file is inside a zip file. we have to extract it first, because in kaggle we can't read the zip file directly because it is private!!!
# df = pd.read_csv("datasets/daniilmiheev/top-spotify-podcasts-daily-updated/top_podcasts.csv")
## For this reason we used this metjod to get data from this csv format file!!!


import zipfile

with zipfile.ZipFile("top-spotify-podcasts-daily-updated.zip", 'r') as zip_ref:
    zip_ref.extractall("datasets/")

df = pd.read_csv("datasets/top_podcasts.csv")

# check for duplicates
item0 = df.shape[0]
df = df.drop_duplicates()
item1 = df.shape[0]
# print(f"There are {item0-item1} duplicates found in the dataset")

# print(df.shape, df.columns)

# select rating as np.log10(max_rank/x)
max_rank = df['rank'].max()
df['rating'] = df['rank'].apply(lambda x: np.log10(max_rank/x))

# log10-transform and bin duration and total eposodes
def log10_transform_bin(x):
    try:
        res = np.log10(x)
        return str(round(round(res*2)/2,1))
    except:
        return 'None'
    

df['log10_duration'] = df['duration_ms'].apply(log10_transform_bin)
df['log10_episodes'] = df['show.total_episodes'].apply(log10_transform_bin)
    
# extract day of the week from date
df['date'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date'].dt.day_name()
df['month'] = df['date'].dt.month_name()




# select columns
selected_cols = ['rating', 'day_of_week', 'month', 'region', 'show.publisher', 
                 'log10_duration', 'log10_episodes', 'explicit', 'is_externally_hosted',
                 'is_playable', 'language', 'show.explicit', 'show.is_externally_hosted',
                 'show.media_type', ]


df = df[selected_cols]


# check for missing values
missing_values = df.isnull().sum()
# print(missing_values)


# print(df.head(10))

# df_audio = df[df["show.media_type"] == "audio"]

# print(df_audio.head(10))


# print(df_audio["show.publisher"].value_counts(ascending=False) )

# with "T" we transpoze the data frame!!!!!
sample_5 = df.sample(5)


# print(sample_5)

columns = df.columns

# print(columns)

# print(df.sample(5).T)



# Select the main label.
main_label = 'rating'


# Set up a rare label encoder for selected columns.
for col in df.columns:
    if col != main_label:
        ## print(f"Label encode {col}")
        df[col] = df[col].fillna('None').astype(str)
        encoder = RareLabelEncoder(n_categories=1, max_n_categories=100, replace_with='Other', tol=20.0 / df.shape[0])
        df[col] = encoder.fit_transform(df[[col]])

"""print(df.shape)  # Print the shape of the resulting DataFrame."""
# df.sample(10).T  # Display a sample of 10 rows, transposed for easier readability.





#################### Machine Learning Part-Section ####################



# Initialize data

# Extract the values of the 'main_label' column and reshape it into a 1D array as 'y'
y = df[main_label].values.reshape(-1,)

# Create the feature matrix 'X' by dropping the 'main_label' column from the DataFrame 'df'
X = df.drop([main_label], axis=1)


# Identify categorical columns in the DataFrame 'df'
# These columns contain non-numeric data, numerik olmayan verileri içerir.
cat_cols = df.select_dtypes(include=['object']).columns

# Create a list of indices for categorical columns in the feature matrix 'X'
cat_cols_idx = [list(X.columns).index(c) for c in cat_cols]


# Split the data into training and testing sets *********** IMPORTANT ***********

# - 'X_train' and 'y_train' will contain the training features and labels, respectively
# - 'X_test' and 'y_test' will contain the testing features and labels, respectively
# The split is done with a 50% test size, a random seed of 0, and stratification based on the selected column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0, stratify=df[['language']])


# Print the dimensions of the training and testing sets
# This provides insight into the sizes of the datasets

"""print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)"""


# Initialize the training and testing data pools using CatBoost's Pool class
train_pool = Pool(X_train, 
                  y_train, 
                  cat_features=cat_cols_idx)  # Create a training data pool with categorical features

test_pool = Pool(X_test,
                 y_test,
                 cat_features=cat_cols_idx)  # Create a testing data pool with categorical features



# Specify the training parameters for the CatBoostRegressor model
model = CatBoostRegressor(iterations=500,    # Number of boosting iterations
                          depth=9,           # Maximum depth of trees in the ensemble
                          verbose=0,         # Set verbosity level to 0 (no output during training)
                          learning_rate=0.000003,  # Learning rate for gradient boosting
                          early_stopping_rounds=100, # Early stopping rounds
                          loss_function='RMSE')  # Loss function to optimize (Root Mean Squared Error)

# Train the CatBoostRegressor model on the training data,   Eğitim verilerini kullanarak modeli eğitiyoruz
model.fit(train_pool, eval_set=test_pool)


model.save_model("trained_model.cbm")

# Make predictions using the trained model on both the training and testing data
y_train_pred = model.predict(train_pool)  # Predictions on the training data
y_test_pred = model.predict(test_pool)    # Predictions on the testing data


# Print the rounded RMSE scores

"""print(f"RMSE score for train {round(rmse_train, 3)} points, and for test {round(rmse_test, 3)} points")"""


shap.initjs()

# Create a TreeExplainer object for the 'model' (assumes 'model' is a tree-based model like a Random Forest or XGBoost)
ex = shap.TreeExplainer(model)

# Calculate SHAP values for the 'X_test' data using the TreeExplainer
shap_values = ex.shap_values(X_test)

# Generate a summary plot to visualize the impact of features on model predictions
"""shap.summary_plot(shap_values, X_test)
plt.show()"""





