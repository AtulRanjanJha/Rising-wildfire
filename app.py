from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import logging
from geopy.geocoders import Nominatim

app = Flask(__name__, template_folder='templates')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model and scaler
try:
    with open("forest_fire_model.pkl", "rb") as model_file:
        model, scaler = pickle.load(model_file)
    logger.info("ML model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Initialize geocoder
geolocator = Nominatim(user_agent="forest_fire_app")

def get_weather_data(lat, lon):
    """Fetch current weather data from Open-Meteo API"""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&"
        f"current=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation&"
        f"timezone=auto"
    )
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        current = data.get('current', {})
        return {
            'temp': current.get('temperature_2m', 20),
            'rh': current.get('relative_humidity_2m', 50),
            'wind': current.get('wind_speed_10m', 5) * 3.6,  # Convert m/s to km/h
            'rain': current.get('precipitation', 0)
        }
    except Exception as e:
        logger.error(f"Weather API error: {str(e)}")
        return {
            'temp': 20,
            'rh': 50,
            'wind': 10,
            'rain': 0
        }

def geocode_location(location_str):
    """Convert location string to coordinates"""
    try:
        if "," in location_str:
            lat, lon = map(float, location_str.split(","))
            return lat, lon
        
        location = geolocator.geocode(location_str)
        if location:
            return location.latitude, location.longitude
        return (28.6139, 77.2090)  # Default to Delhi if not found
    except Exception as e:
        logger.error(f"Geocoding error: {str(e)}")
        return (28.6139, 77.2090)

def calculate_fire_indices(temp, rh, wind, rain):
    ffmc = (0.4 * temp) + (0.2 * wind) - (0.5 * rh) + 60 - (1.5 * rain)
    ffmc = max(18, min(101, ffmc))

    dmc = (0.2 * temp) + (0.3 * wind) + (0.5 * rh) - rain
    dmc = max(0, min(100, dmc))

    dc = (0.3 * temp) + (0.4 * rh) + (0.2 * wind) + 50 - (0.5 * rain)
    dc = max(0, min(500, dc))

    isi = 0.08 * ffmc * (1 + (wind / 10))
    isi = max(0, min(50, isi))

    bui = (dmc + dc) / 2
    fwi = (isi + bui) / 2

    return {
        'ffmc': round(ffmc, 2),
        'dmc': round(dmc, 2),
        'dc': round(dc, 2),
        'isi': round(isi, 2),
        'bui': round(bui, 2),
        'fwi': round(fwi, 2)
    }


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_location', methods=['POST'])
def predict_location():
    try:
        location = request.json.get('location', '').strip()
        if not location:
            return jsonify({'error': 'Location is required'}), 400
        
        lat, lon = geocode_location(location)
        weather = get_weather_data(lat, lon)
        
        indices = calculate_fire_indices(
            weather['temp'],
            weather['rh'],
            weather['wind'],
            weather['rain']
        )
        
        # Prepare features for model prediction
        features = [
            weather['temp'],
            weather['rh'],
            weather['wind'],
            weather['rain'],
            indices['ffmc'],
            indices['dmc'],
            indices['dc'],
            indices['isi'],
            indices['bui'],
            indices['fwi']
        ]
        
        # Make prediction
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)[0]
        
        return jsonify({
            'prediction': 'fire' if prediction == 1 else 'not fire',
            'temp': round(weather['temp'], 1),
            'rh': int(weather['rh']),
            'wind': round(weather['wind'], 1),
            'rain': round(weather['rain'], 1),
            'lat': lat,
            'lon': lon,
            'indices': indices
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)