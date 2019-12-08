from fastapi import FastAPI
from pydantic import BaseModel
from flask import Flask
from ImageEmotionPredictor import EmotionDetector
import numpy as np
import cv2
import base64


class Image(BaseModel):
    encoded_image: str

#app = Flask(__name__)
app = FastAPI()
detector = EmotionDetector()


@app.post('/predict/image')
def predict_image(image: Image):
    print (image)
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
    input_image = get_input_image(image.encoded_image)
    return detector.predict_live_emotion(input_image)

def get_input_image(encoded_image):
    input_image = base64.b64decode(encoded_image)
    input_image = np.asarray(bytearray(input_image), dtype='uint8')
    return cv2.imdecode(input_image, cv2.IMREAD_COLOR)


#if __name__ == '__main__':
#    app.run(host='0.0.0.0', port=5000, debug=True)

