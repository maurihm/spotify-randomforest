import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

filepath = '/home/maurihm_/Documentos/Adsoftsito/pythonPractica/spotify-tracks.csv'

print("Cargando el dataset...")
spotify_data = pd.read_csv(filepath)

spotify_data = spotify_data.dropna(axis=0)

y = spotify_data.popularity
features = [
    'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
    'tempo', 'time_signature'
]
X = spotify_data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

print("Entrenando el modelo Random Forest...")
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)

print("Generando predicciones...")
spotify_preds = forest_model.predict(val_X)

print("--- Random Forest Regressor ---")
print("Mean Absolute Error: %.2f" % mean_absolute_error(val_y, spotify_preds))
