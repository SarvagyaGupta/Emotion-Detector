"""
Represents the model of the emotion detector
"""


from keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Dropout, MaxPooling2D
from keras.models import Sequential, model_from_json
from keras.regularizers import l2
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import keras
import tensorflow as tf
from DataLoader import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Include this if plt not saving graphs
# import os
#
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

# config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
# sess = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(sess)

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

num_features = 64
num_labels = 7
batch_size = 128
epochs = 150
width, height = 48, 48

data = DataLoader(num_labels)
train, validation, test = data.train, data.validation, data.test

train_pixels, train_emotions = [image.pixels for image in train], [image.label for image in train]
validation_pixels, validation_emotions = [image.pixels for image in validation], [image.label for image in validation]
test_pixels, test_emotions = [image.pixels for image in test], [image.label for image in test]

# Creates a model that stacks layers on top of each other
model = Sequential()

# Input shape should be 3D tensor
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1)
                 , kernel_regularizer=l2(0.01)))
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)))

# Helps avoid overfitting and removes dependency of layers on each other
model.add(BatchNormalization())

# Helps avoid overfitting and reduces dimensionality
model.add(MaxPooling2D())

# Dropping nodes randomly to speed up training and avoid overfitting
model.add(Dropout(0.5))

model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.5))

model.add(Conv2D(3 * num_features, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Conv2D(3 * num_features, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.5))


# Converts the NN into 1D
model.add(Flatten())
# Maps the features learnt from layers above to the labels
model.add(Dense(num_labels, activation='softmax'))

# categorical_crossentropy is used when there is one correct result
# Adam loss is basically SGD
model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['accuracy'])

# Running the model
history = model.fit(np.array(train_pixels), np.array(train_emotions)
                    , validation_data=(np.array(validation_pixels), np.array(validation_emotions))
                    , batch_size=batch_size, epochs=epochs, shuffle=True
                    , callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=20)])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train Set', 'Validation Set'], loc='upper left')
plt.savefig('../model/loss_curr_best_175')
plt.close()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train Set', 'Validation Set'], loc='upper left')
plt.savefig('../model/accuracy_curr_best_175')
plt.close()
# Saving the model
saved_model = model.to_json()
with open("../model/emotion_detector_plot_curr_best_175.json", "w") as json_file:
    json_file.write(saved_model)
model.save_weights("../model/emotion_detector_plot_curr_best_175.h5")

emotion_file = open("../model/emotion_detector_plot_curr_best_175.json", 'r')
loaded_model_json = emotion_file.read()
emotion_file.close()
modeled_emotion = model_from_json(loaded_model_json)

# Load the weights
modeled_emotion.load_weights("../model/emotion_detector_plot_curr_best_175.h5")

print ('Model loaded... Calculating the accuracy on the test dataset...')

# Working on test data
predicted_emotions = modeled_emotion.predict(np.asarray(test_pixels))

count = 0
for i in range(len(predicted_emotions)):
    if np.argmax(predicted_emotions[i]) == np.argmax(test_emotions[i]):
        count += 1

print ('Accuracy = ', 100. * count / len(predicted_emotions))
