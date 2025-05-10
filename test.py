import requests
import json

# Replace with the local URL where your Flask app is running
URL = "http://127.0.0.1:5000/predict"

# Image file path (use raw string or forward slashes)
image_path = "C:\\Users\\kpkir\\Downloads\\potatoblight.jpg"
#sensor example data
sensor_data = {
    "temperature": 28.5,
    "humidity": 65,
    "soil_moisture": 450,
    "light": 300
}

# Prepare request
files = {
    'image': open(image_path, 'rb')
}
data = {
    'sensor_data': json.dumps(sensor_data)
}

# Send POST request
response = requests.post(URL, files=files, data=data)

# Print response
print("Status Code:", response.status_code)
print("Response JSON:")
print(json.dumps(response.json(), indent=2))
