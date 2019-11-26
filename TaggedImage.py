"""
TaggedImage is representation of one image that represents an emotion
"""

import numpy as np


class TaggedImage:

    def __init__(self, csv_row):
        self.label = csv_row['emotion']
        self.pixels = self.reshape_pixels(csv_row['pixels'])

    def reshape_pixels(self, pixels):
        return np.array(pixels.split()).reshape(48, 48)
