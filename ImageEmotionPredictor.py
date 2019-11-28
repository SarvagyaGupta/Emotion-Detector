"""
Predicts the emotion from an image
"""

from keras.models import model_from_json
import numpy as np
import cv2

# Load the model
emotion_file = open('emotion_detector.json', 'r')
loaded_model_json = emotion_file.read()
emotion_file.close()
modeled_emotion = model_from_json(loaded_model_json)

# Load the weights
modeled_emotion.load_weights('emotion_detector.h5')

labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load image from webcam
image = cv2.imread("./test/test4.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face.detectMultiScale(gray, 1.3, 10)

for (x, y, w, h) in faces:
    roi_gray = gray[y:y + h, x:x + w]
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
    cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # predicting the emotion
    yhat = modeled_emotion.predict(cropped_img)
    cv2.putText(image, labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
    print("Emotion: " + labels[int(np.argmax(yhat))])

cv2.imwrite('Emotion.jpg', image)
cv2.waitKey()
