import requests

url = 'http://34.94.140.117:5000/predict'
response = requests.get(url)

print response.json()
