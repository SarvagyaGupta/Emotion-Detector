import requests
import json
import cv2

url = 'http://34.94.140.117:5000/predict'
image = cv2.imread("../test/test.jpg")
_, image_encoded = cv2.imencode('.jpg', image)
response = requests.post(url, data=image_encoded.tostring())

print json.loads(response.text)
