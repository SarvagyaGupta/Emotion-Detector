from fastapi import FastAPI
from pydantic import BaseModel
from flask import Flask
from ImageEmotionPredictor import EmotionDetector
import numpy as np
import cv2
import base64
import settings
import uuid
import time
import json
import redis


class Image(BaseModel):
    encoded_image: str

app = FastAPI()
detector = EmotionDetector
db = redis.StrictRedis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB)


@app.post('/predict/image')
def predict_image(image: Image):
    input_image = get_input_image(image.encoded_image)
    predictions = detector.predict_emotion(input_image)
    for prediction in predictions['predictions']:
        detector.draw_face_boundary(input_image, (prediction['top_left_x'], prediction['top_left_y']
                                                  , prediction['width'], prediction['height']), (0, 255, 0))

    _, output_image = cv2.imencode('.jpg', input_image)
    predictions['image'] = base64.b64encode(output_image)

    return predictions


@app.post('/predict/live')
def predict_live(image: Image):
    #input_image = get_input_image(image.encoded_image)
    #_, output_image = cv2.imencode('.jpg', input_image)
    #output_image = base64.b64encode(output_image).decode("utf-8")
    #db = redis.StrictRedis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB)
    print("here")
    output_image = image.encoded_image
    key = str(uuid.uuid4())
    d = {"id": key, "image": output_image}
    db.rpush(settings.IMAGE_QUEUE, json.dumps(d))

    response = None

    while True:
        output_json = db.get(key)

        if output_json is not None:
            response = json.loads(output_json.decode("utf-8"))
            db.delete(key)
            break

        time.sleep(settings.CLIENT_SLEEP)

    return response


def get_input_image(encoded_image):
    input_image = base64.b64decode(encoded_image)
    input_image = np.asarray(bytearray(input_image), dtype='uint8')
    return cv2.imdecode(input_image, cv2.IMREAD_COLOR)
