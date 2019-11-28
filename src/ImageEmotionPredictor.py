"""
Predicts the emotion from an image
"""

from keras.models import model_from_json
import numpy as np
import cv2


class EmotionDetector:

    def __init__(self):
        # Load the model
        emotion_file = open('model/emotion_detector.json', 'r')
        loaded_model_json = emotion_file.read()
        emotion_file.close()
        self.modeled_emotion = model_from_json(loaded_model_json)

        # Load the weights
        self.modeled_emotion.load_weights('model/emotion_detector.h5')

        self.labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def flask_test(self):
        return 'Done'

    def predict_emotion(self, image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face.detectMultiScale(gray, 1.3, 10)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # predicting the emotion
            yhat = self.modeled_emotion.predict(cropped_img)
            cv2.putText(image, self.labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
            print("Emotion: " + self.labels[int(np.argmax(yhat))])

        cv2.imwrite('Emotion.jpg', image)
        cv2.waitKey()
