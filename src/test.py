from threading import Thread
import requests
import time
import cv2
import base64

URL = "http://34.94.140.117:5000/predict/live"
IMAGE_PATH = "../2014 - 1.jpg"

# initialize the number of requests for the stress test along with
# the sleep amount between requests
NUM_REQUESTS = 500
SLEEP_COUNT = 0.05

def call_predict_endpoint(n):
    # load the input image and construct the payload for the request
    with open(IMAGE_PATH, 'rb') as f:
        image = f.read()

        #image = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    #_, image_encoded = cv2.imencode('.jpg', image)
    #image_encoded = base64.b64encode(image_encoded)
        body = {"encoded_image": image}
        r = requests.post(URL, files=body).json()

        print n, r


# loop over the number of threads
for i in range(0, NUM_REQUESTS):
    # start a new thread to call the API
    t = Thread(target=call_predict_endpoint, args=(i,))
    t.daemon = True
    t.start()
    time.sleep(SLEEP_COUNT)

# insert a long sleep so we can wait until the server is finished
# processing the images
time.sleep(300)
