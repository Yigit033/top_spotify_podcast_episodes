from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from catboost import Pool, CatBoostRegressor

app = Flask(__name__)

import zipfile

# Flask server için öncelikle modeli ve veriyi yükle


with zipfile.ZipFile("top-spotify-podcasts-daily-updated.zip", 'r') as zip_ref:
    zip_ref.extractall("datasets/")

df = pd.read_csv("datasets/top_podcasts.csv")

"""df = pd.read_csv("datasets/top_podcasts.csv")"""

model = CatBoostRegressor()
model.load_model("trained_model.cbm")  # Eğitilmiş modeli yükle

# @app.route('/', methods=['POST'])

@app.route('/')

def predict():
    # Kullanıcıdan gelen veriyi al
    data = request.json
    
    # Veriyi işleme (örneğin, kullanıcı sorgusu ile ilgili veriyi çek)
    podcast_data = data['podcast_info']  # Kullanıcıdan podcast bilgisi al
    
    # Model ile tahmin yap (örneğin podcast rating tahmini)
    prediction = model.predict(podcast_data)  
    
    # Kullanıcıya sonuçları ilet
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)

