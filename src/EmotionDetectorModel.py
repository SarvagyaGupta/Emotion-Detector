"""
Represents the model of the emotion detector
"""

from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.models import Sequential
from keras.regularizers import l2
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from DataLoader import DataLoader
import numpy as np


num_features = 64
num_labels = 7
batch_size = 128
epochs = 100
width, height = 48, 48

data = DataLoader(num_labels)
train, test = data.train, data.test

train_pixels, train_emotions = [image.pixels for image in train], [image.label for image in train]
test_pixels, test_emotions = [image.pixels for image in test], [image.label for image in test]

# Creates a model that stacks layers on top of each other
model = Sequential()

# Input shape should be 3D tensor
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), kernel_regularizer=l2(0.01)))
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01), padding='same'))

# Helps avoid overfitting and removes dependency of layers on each other
model.add(BatchNormalization())

# Helps avoid overfitting and reduces dimensionality
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

# Dropping nodes randomly to speed up training and avoid overfitting
model.add(Dropout(0.3))

model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

# Converts the NN into 1D
model.add(Flatten())

# Maps the features learnt from layers above to the labels
model.add(Dense(num_labels, activation='softmax'))

# categorical_crossentropy is used when there is one correct result
# Adam loss is basically SGD
model.compile(optimizer=Adam(), loss=categorical_crossentropy)

# Running the model
model.fit(np.array(train_pixels), np.array(train_emotions), batch_size=batch_size, epochs=epochs, shuffle=True)

# Saving the model
saved_model = model.to_json()
with open("../model/emotion_detector_3.json", "w") as json_file:
    json_file.write(saved_model)
model.save_weights("../model/emotion_detector_3.h5")
