import sys
import requests

# URL base configurable por argumento de línea de comandos (ej. para probar en Render)
base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
endpoint = f"{base_url.rstrip('/')}/score"

body = {
    "duration_ms": 230666,
    "danceability": 0.676,
    "energy": 0.461,
    "key": 1,
    "loudness": -6.746,
    "mode": 0,
    "speechiness": 0.143,
    "acousticness": 0.0322,
    "instrumentalness": 0.000001,
    "liveness": 0.358,
    "valence": 0.715,
    "tempo": 87.917,
    "time_signature": 4
}

print(f"Enviando petición a: {endpoint}")
try:
    response = requests.post(url=endpoint, json=body)
    print(f"Código HTTP: {response.status_code}")
    print("Respuesta:", response.json())
except Exception as e:
    print(f"Error al conectar con el servidor: {e}")
