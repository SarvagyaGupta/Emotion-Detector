from flask import Flask, jsonify, request
from ImageEmotionPredictor import EmotionDetector
import cv2
import json

app = Flask(__name__)

detector = EmotionDetector()


@app.route('/predict', methods=['POST'])
def predict():
    r = request
    predictions = detector.predict_emotion(r.data)

    response = json.dumps(predictions)
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
