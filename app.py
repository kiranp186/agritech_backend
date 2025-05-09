from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
import requests
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load model
try:
    model = tf.keras.models.load_model('model.h5')
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Define class names for prediction
CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___Healthy"
]

# Weather API configuration
WEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"

# Utility: Check if file type is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Main prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500

    # 1. Parse sensor data (optional)
    sensor_data = {}
    if 'sensor_data' in request.form:
        try:
            sensor_data = json.loads(request.form['sensor_data'])
        except json.JSONDecodeError:
            return jsonify({"status": "error", "message": "Invalid sensor data format"}), 400

    # 2. Handle image upload
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected image file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "Invalid file type"}), 400

    temp_path = None
    try:
        # Save and process the image
        filename = secure_filename(file.filename)
        os.makedirs('uploads', exist_ok=True)
        temp_path = os.path.join('uploads', filename)
        file.save(temp_path)

        img = Image.open(temp_path).convert('RGB').resize((256, 256))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # 3. Predict
        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        # 4. Get weather data
        weather_data = get_weather_data()

        # 5. Analyze sensor data
        sensor_insights = analyze_sensor_data(sensor_data)

        # 6. Generate final response
        response = {
            "status": "success",
            "disease": predicted_class,
            "confidence": confidence,
            "sensor_data": sensor_data,
            "sensor_analysis": sensor_insights,
            "weather_data": weather_data,
            "recommendation": get_remedy(predicted_class)
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# Analyze sensor data and generate insights
def analyze_sensor_data(data):
    advice = []
    temp = data.get("temperature")
    humidity = data.get("humidity")
    soil_moisture = data.get("soil_moisture")

    if temp is not None:
        if temp > 35:
            advice.append("Temperature is high — consider shade or irrigation.")
        elif temp < 15:
            advice.append("Temperature is low — crop growth might slow.")

    if humidity is not None:
        if humidity > 80:
            advice.append("High humidity may promote fungal diseases.")
        elif humidity < 30:
            advice.append("Low humidity may cause dehydration.")

    if soil_moisture is not None:
        if soil_moisture < 200:
            advice.append("Soil is dry — consider watering.")
        elif soil_moisture > 800:
            advice.append("Soil may be waterlogged — ensure proper drainage.")

    return advice

# Fetch current weather data (defaults to Bangalore)
def get_weather_data(lat=12.97, lon=77.59):
    if not WEATHER_API_KEY:
        return None
    try:
        params = {
            'lat': lat,
            'lon': lon,
            'appid': WEATHER_API_KEY,
            'units': 'metric'
        }
        response = requests.get(WEATHER_API_URL, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        return {
            'temperature': data.get('main', {}).get('temp'),
            'humidity': data.get('main', {}).get('humidity'),
            'conditions': data.get('weather', [{}])[0].get('main')
        }
    except requests.RequestException:
        return None

# Provide remedy based on disease class
def get_remedy(disease):
    remedies = {
        "Potato___Early_blight": (
            "1. Remove infected leaves\n"
            "2. Apply copper-based fungicide\n"
            "3. Improve air circulation"
        ),
        "Potato___Late_blight": (
            "1. Apply fungicides with chlorothalonil\n"
            "2. Remove and destroy infected plants\n"
            "3. Avoid overhead watering"
        ),
        "Potato___Healthy": "No treatment needed. Maintain good growing conditions."
    }
    return remedies.get(disease, "Consult an agricultural expert for diagnosis and treatment.")

# Run the Flask server
if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    print("✅ Weather API Key Loaded:", WEATHER_API_KEY is not None)
    app.run(host='0.0.0.0', port=5000, debug=True)
