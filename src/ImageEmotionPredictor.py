"""
Predicts the emotion from an image
"""

import heapq

import cv2
import numpy as np
from keras.models import model_from_json
import base64
import tensorflow as tf
from tensorflow import keras


config = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)
session = tf.Session(config=config)
keras.backend.set_session(session)


class EmotionDetector:

    def __init__(self):
        # Load the model
        emotion_file = open('../model/current_best.json', 'r')
        loaded_model_json = emotion_file.read()
        emotion_file.close()
        self.modeled_emotion = model_from_json(loaded_model_json)
        self.modeled_emotion._make_predict_function()

        # Load the weights
        self.modeled_emotion.load_weights('../model/current_best.h5')

        self.labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def predict_emotion(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_classifier = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        predictions = {'predictions': []}

        for face in faces:
            predictions['predictions'].append(self.__predict_face_emotion(gray, face))

        return predictions

    def predict_live_emotion(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_classifier = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for face in faces:
            self.__predict_live_emotion(image, gray, face)

        _, output_image = cv2.imencode('.jpg', image)
        output_image = base64.b64encode(output_image)
        return {'image': output_image}


    def __predict_live_emotion(self, image, gray, face):
        x, y, w, h = face
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)

        with session.as_default():
            with session.graph.as_default():
 
                prediction = self.modeled_emotion.predict(cropped_img)
                predicted_emotion = self.labels[int(np.argmax(prediction))]

                self.draw_face_boundary(image, face, (0, 255, 0))
                self.write_emotion(image, face, predicted_emotion, (0, 255, 0)) 


    def __predict_face_emotion(self, gray, face):
        x, y, w, h = face
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        
        with session.as_default():
            with session.graph.as_default():
                # predict the probability
                probability = self.modeled_emotion.predict_proba(cropped_img)
                probab = []
                for i in probability[0]:
                    probab.append(float('{:f}'.format(i)))

                # index of the top 3 emotions
                photo_indices = heapq.nlargest(3, range(len(probab)), probab.__getitem__)

                # percentage of that emotion
                percentage = [100 * probab[index] for index in photo_indices]
                percentage = [round(percent, 2) for percent in percentage]

                # labels of those percentages
                emotions = [self.labels[index] for index in photo_indices]

                return {'top_left_x': x.item(), 'top_left_y': y.item(), 'height': h.item(), 'width': w.item()
                        , 'emotions': emotions, 'emotion_percentages': percentage}


    def draw_face_boundary(self, image, face, color):
        x, y, w, h = face
        cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=color, thickness=2)


    def write_emotion(self, image, face, emotion, color):
        x, y, w, h = face
        cv2.putText(image, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)
