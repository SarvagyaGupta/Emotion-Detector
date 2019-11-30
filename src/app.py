from flask import Flask, request
from ImageEmotionPredictor import EmotionDetector
import numpy as np
import cv2

app = Flask(__name__)

detector = EmotionDetector()


@app.route('/predict', methods=['POST'])
def predict():
    data = request.files['image'].read()
    input_image = np.asarray(bytearray(data), dtype='uint8')
    input_image = cv2.imdecode(input_image, cv2.IMREAD_COLOR)

    predictions = detector.predict_emotion(input_image)
    for prediction in predictions['predictions']:
        detector.draw_face_boundary(input_image, (prediction['top_left_x'], prediction['top_left_y']
                                                  , prediction['width'], prediction['height']), (0, 255, 0))

    predictions['image'] = cv2.imencode('.jpg', input_image).read().encode('base64')
    return predictions


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

