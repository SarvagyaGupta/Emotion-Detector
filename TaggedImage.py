"""
TaggedImage is representation of one image that represents an emotion
"""

import numpy as np


class TaggedImage:

    def __init__(self, csv_row):
        self.label = csv_row['emotion']
        self.pixels = np.array(csv_row['pixels'].split()).reshape(48, 48)
