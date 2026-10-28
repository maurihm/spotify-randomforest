#!/bin/bash
# Script de prueba rápida con curl para evaluar la API

# Si le pasas la URL de Render como argumento la usa, sino usa localhost por defecto:
# Ejemplo: ./curl_test.sh https://tu-servicio.onrender.com
URL="${1:-http://127.0.0.1:8000}"
ENDPOINT="${URL%/}/score"

echo "Probando endpoint: $ENDPOINT"
echo "--------------------------------------------------"

curl -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'

echo ""
echo "--------------------------------------------------"
