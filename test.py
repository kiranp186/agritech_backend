import requests
import json

image_path = "C:/Users/kpkir/Downloads/sick_leaf.jpg"

sensor_data = {
    "temperature": 32.5,
    "humidity": 70,
    "soil_moisture": 300
}

response = requests.post(
    "http://localhost:5000/predict",
    files={"image": open(image_path, "rb")},
    data={"sensor_data": json.dumps(sensor_data)}
)

print(response.json())
