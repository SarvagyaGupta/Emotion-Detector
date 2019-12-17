from fastapi import FastAPI
from pydantic import BaseModel
from flask import Flask
from ImageEmotionPredictor import EmotionDetector
import numpy as np
import cv2
import base64


class Image(BaseModel):
    encoded_image: str

app = FastAPI()
detector = EmotionDetector()


@app.post('/predict')
def predict_image(image: Image):
    input_image = get_input_image(image.encoded_image)
    return detector.predict_emotion(input_image)


def get_input_image(encoded_image):
    input_image = base64.b64decode(encoded_image)
    input_image = np.asarray(bytearray(input_image), dtype='uint8')
    return cv2.imdecode(input_image, cv2.IMREAD_COLOR)

