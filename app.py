from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
import requests
import os
import json
from dotenv import load_dotenv
import urllib.request

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Model settings
MODEL_PATH = "model_quant.tflite"
MODEL_URL = os.getenv("MODEL_URL")
interpreter = None

# Download model if not present
def download_model():
    if not os.path.exists(MODEL_PATH) and MODEL_URL:
        try:
            print(f"Downloading model from {MODEL_URL}...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("✅ Model downloaded successfully.")
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
    elif not MODEL_URL:
        print("❌ MODEL_URL not set!")

download_model()

# Load TFLite model
if os.path.exists(MODEL_PATH):
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        print("✅ TFLite model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load TFLite model: {e}")

# Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___Healthy"
]
WEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"

# Helper: allowed file check
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Main endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if not interpreter:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500

    # Log the incoming request
    print("📥 Received prediction request")

    sensor_data = {}
    if 'sensor_data' in request.form:
        try:
            sensor_data = json.loads(request.form['sensor_data'])
            print(f"📊 Sensor data received: {sensor_data}")
        except json.JSONDecodeError:
            print("❌ Invalid sensor data format")
            return jsonify({"status": "error", "message": "Invalid sensor data format"}), 400

    # Process request with or without image
    has_image = 'image' in request.files and request.files['image'].filename != ''
    
    if has_image:
        file = request.files['image']
        
        if not allowed_file(file.filename):
            print(f"❌ Invalid file type: {file.filename}")
            return jsonify({"status": "error", "message": "Invalid file type"}), 400

        temp_path = None
        try:
            filename = secure_filename(file.filename)
            os.makedirs('uploads', exist_ok=True)
            temp_path = os.path.join('uploads', filename)
            file.save(temp_path)
            print(f"✅ Image saved temporarily: {temp_path}")

            img = Image.open(temp_path).convert('RGB').resize((256, 256))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Prepare tensors for TFLite prediction
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]

            predicted_class = CLASS_NAMES[np.argmax(output_data)]
            confidence = float(np.max(output_data))
            print(f"🔍 Image analysis result: {predicted_class} with {confidence:.2f} confidence")

        except Exception as e:
            print(f"❌ Error processing image: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
                print("🗑️ Temporary image file removed")
    else:
        print("ℹ️ No image provided, skipping disease detection")
        predicted_class = "No_image_provided"
        confidence = 0.0

    # Always process sensor data
    weather_data = get_weather_data()
    sensor_insights = analyze_sensor_data(sensor_data)

    response = {
        "status": "success",
        "sensor_data": sensor_data,
        "sensor_analysis": sensor_insights,
        "weather_data": weather_data,
    }
    
    # Add disease prediction if image was provided
    if has_image:
        response.update({
            "disease": predicted_class,
            "confidence": confidence,
            "recommendation": get_remedy(predicted_class)
        })
        
    print(f"✅ Processed request successfully")
    return jsonify(response)

# Helper: sensor analysis with light sensor support
def analyze_sensor_data(data):
    advice = []
    temp = data.get("temperature")
    humidity = data.get("humidity")
    soil_moisture = data.get("soil_moisture")
    light = data.get("light")
    
    # Log received sensor values
    print(f"🌡️ Temperature: {temp}°C") if temp is not None else print("❌ No temperature data")
    print(f"💧 Humidity: {humidity}%") if humidity is not None else print("❌ No humidity data")
    print(f"🌱 Soil Moisture: {soil_moisture}") if soil_moisture is not None else print("❌ No soil moisture data")
    print(f"☀️ Light Level: {light}") if light is not None else print("❌ No light data")

    if temp is not None:
        if temp > 35:
            advice.append("Temperature is high (above 35°C) — consider shade or irrigation to cool plants.")
        elif temp < 15:
            advice.append("Temperature is low (below 15°C) — crop growth might slow down.")
        elif 15 <= temp <= 25:
            advice.append("Temperature is in optimal range for potato growth.")

    if humidity is not None:
        if humidity > 80:
            advice.append("High humidity (above 80%) may promote fungal diseases like late blight.")
        elif humidity < 30:
            advice.append("Low humidity (below 30%) may cause plant dehydration.")
        else:
            advice.append("Humidity levels are moderate.")

    if soil_moisture is not None:
        if soil_moisture < 300:
            advice.append("Soil is very dry — water plants immediately.")
        elif soil_moisture < 500:
            advice.append("Soil is becoming dry — consider watering soon.")
        elif soil_moisture > 800:
            advice.append("Soil may be waterlogged — ensure proper drainage to prevent root rot.")
        else:
            advice.append("Soil moisture is in a good range.")

    # Add light level analysis
    if light is not None:
        if light < 200:
            advice.append("Light level is very high — plants are getting strong sunlight.")
        elif light < 500:
            advice.append("Light level is good for plant growth.")
        elif light > 800:
            advice.append("Light level is low — plants may not be getting enough sunlight for optimal photosynthesis.")

    return advice

# Helper: weather API
def get_weather_data(lat=12.97, lon=77.59):
    if not WEATHER_API_KEY:
        print("❌ Weather API key not set, skipping weather data")
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
        weather_data = {
            'temperature': data.get('main', {}).get('temp'),
            'humidity': data.get('main', {}).get('humidity'),
            'conditions': data.get('weather', [{}])[0].get('main')
        }
        print(f"🌤️ Weather data retrieved: {weather_data}")
        return weather_data
    except requests.RequestException as e:
        print(f"❌ Weather API error: {str(e)}")
        return None

# Helper: remedy with expanded recommendations
def get_remedy(disease):
    remedies = {
        "Potato___Early_blight": (
            "1. Remove and destroy infected leaves immediately\n"
            "2. Apply copper-based fungicide every 7-10 days\n"
            "3. Improve air circulation between plants\n"
            "4. Water at the base of plants to keep foliage dry\n"
            "5. Rotate crops in the future to prevent disease buildup"
        ),
        "Potato___Late_blight": (
            "1. Apply fungicides with chlorothalonil or copper-based products\n"
            "2. Remove and destroy infected plants completely\n"
            "3. Avoid overhead watering and irrigate in the morning\n"
            "4. Increase plant spacing to improve air circulation\n"
            "5. Consider harvesting early if disease is widespread"
        ),
        "Potato___Healthy": (
            "Your potato plants appear healthy. Maintain good growing conditions:\n"
            "1. Continue balanced watering\n"
            "2. Apply fertilizer according to growth stage\n"
            "3. Monitor for pest activity\n"
            "4. Maintain good air circulation"
        ),
        "No_image_provided": "No image was provided for disease detection. Monitoring sensor data only."
    }
    return remedies.get(disease, "Consult an agricultural expert for diagnosis and treatment.")

@app.route("/")
def home():
    return "✅ AgriTech Flask Backend is running! Use /predict endpoint to analyze plant health."

# Logging routes
@app.route("/logs")
def check_logs():
    return "Check the application logs on your hosting platform for detailed information."

# Start app
if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    print("✅ AgriTech Backend Starting...")
    print("✅ Weather API Key Loaded:", WEATHER_API_KEY is not None)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)