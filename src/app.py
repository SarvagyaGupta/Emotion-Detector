from flask import Flask, request
from ImageEmotionPredictor import EmotionDetector
import numpy as np
import cv2

app = Flask(__name__)

detector = EmotionDetector()


@app.route('/predict', methods=['POST'])
def predict():
    data = request.files['image'].read()
    image = np.asarray(bytearray(data), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return detector.predict_emotion(image)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

