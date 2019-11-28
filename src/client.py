import requests
import json

url = 'http://34.94.140.117:5000/predict'
response = requests.post(url)

print json.loads(response.text)
