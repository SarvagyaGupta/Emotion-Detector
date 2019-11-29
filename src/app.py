from flask import Flask, request
from ImageEmotionPredictor import EmotionDetector
import werkzeug

app = Flask(__name__)

detector = EmotionDetector()


@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    print image_file
    print type(image_file)

    filename = werkzeug.utils.secure_filename(image_file.filename)
    image_file.save(filename)

    return detector.predict_emotion("123")

