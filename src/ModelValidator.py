"""
Validates the model and displays its accuracy
"""

from keras.models import model_from_json
from DataLoader import DataLoader
import numpy as np

num_labels = 7
data = DataLoader(num_labels)
test = data.test

test_pixels, test_emotions = [image.pixels for image in test], [image.label for image in test]

# Load the model
emotion_file = open('model/emotion_detector.json', 'r')
loaded_model_json = emotion_file.read()
emotion_file.close()
modeled_emotion = model_from_json(loaded_model_json)

# Load the weights
modeled_emotion.load_weights('model/emotion_detector.h5')

print 'Model loaded... Making very accurate predictions...'

# Working on test data
predicted_emotions = modeled_emotion.predict(np.asarray(test_pixels))

count = 0
for i in range(len(predicted_emotions)):
    if np.argmax(predicted_emotions[i]) == np.argmax(test_emotions[i]):
        count += 1

print 'Accuracy = ', 100. * count / len(predicted_emotions)
