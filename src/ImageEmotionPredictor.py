"""
Predicts the emotion from an image
"""

from keras.models import model_from_json
import numpy as np
import cv2
import pandas as pd
import heapq
import random
import math
import sys


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

        # reading the celeb tweets
        self.celeb = pd.read_csv("../FinalCelebrityTweets.csv")

    def predict_emotion(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_classifier = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        predictions = {'predictions': []}

        for face in faces:
            predictions['predictions'].append(self.__predict_face_emotion(gray, face))

        return predictions

    def __calculate_closeness(self, x, y):
        total = 0
        for i in range(len(x)):
            total += (x[i] - y[i]) ** 2
        total = math.sqrt(total)
        return total

    def __get_tweet_info(self, data, probabilities, photo_indices):
        result = []
        minimum = sys.maxsize

        for index_label, row_series in data.iterrows():
            temp = [float(x) for x in data.at[index_label, 'probs'][2:-2].split()]
            indices = heapq.nlargest(3, range(len(temp)), temp.__getitem__)
            e = 0
            for i in indices:
                if i in photo_indices:
                    e += 1
            if e > 1 and photo_indices[0] == indices[0]:
                p = self.__calculate_closeness(temp, probabilities)
                if p < minimum:
                    minimum = p
                    tweet = data.at[index_label, 'content']
                    person = data.at[index_label, 'author']
                    result.append((person, tweet, p))

        if len(result) == 0:
            return "", ""

        result.sort(key=lambda tup: tup[2])
        result = result[:2]
        twe = random.choice(result)
        return twe[0], twe[1]

    def __predict_face_emotion(self, gray, face):
        x, y, w, h = face
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)

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

        celebrity, tweet = self.__get_tweet_info(self.celeb, probab, photo_indices)

        return {'top_left_x': x.item(), 'top_left_y': y.item(), 'height': h.item(), 'width': w.item()
                , 'emotions': emotions, 'emotion_percentages': percentage, 'celebrity': celebrity, 'tweet': tweet}

    def draw_face_boundary(self, image, face, color):
        x, y, w, h = face
        cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=color, thickness=2)
