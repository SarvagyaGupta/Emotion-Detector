"""
Predicts the emotion from an image
"""

from keras.models import model_from_json
import numpy as np
import cv2


class EmotionDetector:

    def __init__(self):
        # Load the model
        emotion_file = open('../model/emotion_detector.json', 'r')
        loaded_model_json = emotion_file.read()
        emotion_file.close()
        self.modeled_emotion = model_from_json(loaded_model_json)
        self.modeled_emotion._make_predict_function()

        # Load the weights
        self.modeled_emotion.load_weights('../model/emotion_detector.h5')

        self.labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def predict_emotion(self, image_encoded_str):
        """

        :param image_encoded_str:
        :return:
        """

        image_encoded = np.fromstring(image_encoded_str, np.uint8)
        image = cv2.imdecode(image_encoded, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_classifier = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
        faces = face_classifier.detectMultiScale(gray, 1.3, 10)

        predictions = {'predictions': []}

        for face in faces:
            # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # cv2.putText(image, self.labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
            predictions['predictions'].append(self.__predict_face_emotion(gray, face))

        return predictions

    def __predict_face_emotion(self, gray, face):
        x, y, w, h = face
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        predicted_emotion = self.labels[int(np.argmax(self.modeled_emotion.predict(cropped_img)))]

        return {'top_left_x': str(x), 'top_left_y': str(y), 'height': str(h), 'width': str(w), 'emotion': str(predicted_emotion)}
