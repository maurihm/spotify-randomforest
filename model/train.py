import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from joblib import dump

# Localizar archivo del dataset
dataset_paths = [
    pathlib.Path('data/spotify-tracks.csv'),
    pathlib.Path(__file__).resolve().parent.parent / 'data' / 'spotify-tracks.csv',
    pathlib.Path(__file__).resolve().parent.parent.parent / 'spotify-tracks.csv',
    pathlib.Path('spotify-tracks.csv'),
    pathlib.Path('../spotify-tracks.csv'),
    pathlib.Path('/home/maurihm_/Documentos/Adsoftsito/Modelos/spotify-tracks.csv'),
    pathlib.Path('/home/maurihm_/Documentos/Adsoftsito/pythonPractica/spotify-tracks.csv')
]

dataset_file = None
for p in dataset_paths:
    if p.exists():
        dataset_file = p
        break

if dataset_file is None:
    raise FileNotFoundError("No se encontró el archivo spotify-tracks.csv")

print(f"Cargando dataset desde: {dataset_file}...")
spotify_data = pd.read_csv(dataset_file)
spotify_data = spotify_data.dropna(axis=0)

y = spotify_data['popularity']
features = [
    'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
    'tempo', 'time_signature'
]
X = spotify_data[features]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

print("Entrenando modelo RandomForestRegressor...")
model = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=1, n_jobs=-1)
model.fit(X_train, y_train)

preds = model.predict(X_val)
mae = mean_absolute_error(y_val, preds)
print(f"Entrenamiento finalizado. MAE en validación: {mae:.2f}")

output_dir = pathlib.Path(__file__).resolve().parent
output_dir.mkdir(parents=True, exist_ok=True)
model_path = output_dir / 'spotify-rf-v1.joblib'

print(f"Guardando modelo en: {model_path}...")
dump(model, model_path, compress=3)
print("¡Modelo guardado exitosamente!")
