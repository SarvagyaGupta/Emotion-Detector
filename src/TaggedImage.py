"""
TaggedImage is representation of one image that represents an emotion
"""

import numpy as np


class TaggedImage:

    def __init__(self, csv_row, num_emotions):
        self.label = [0] * num_emotions
        self.label[csv_row['emotion']] = 1

        self.pixels = [float(x) for x in csv_row['pixels'].split()]
        self.pixels = np.asarray(self.pixels).reshape(48, 48)
        self.pixels = np.expand_dims(self.pixels, -1)
