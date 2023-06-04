import requests

url = "http://0.0.0.0:5000/predict"  # localhost and the defined port + endpoint
body = {"image_path": "sample_image.jpg"}
response = requests.post(url, data=body)
print(response.json())
