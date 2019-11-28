from flask import Flask, jsonify, request
from ImageEmotionPredictor import EmotionDetector
import cv2
import jsonpickle

app = Flask(__name__)

detector = EmotionDetector()


@app.route('/predict', methods=['POST'])
def predict():
    image = cv2.imread("../test/test.jpg")
    _, image_encoded = cv2.imencode('.jpg', image)
    predictions = detector.predict_emotion(image_encoded)

    response = jsonpickle.encode(predictions)
    # response.status_code = 200
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
