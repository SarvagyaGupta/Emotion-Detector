"""

"""

import numpy as np

class TaggedImage:

    def __init__(self, csv_row):
        self.label = csv_row['emotion']
        self.pixels = csv_row['pixels']

    def reshape_pixels(self):
        self.pixels = np.array(self.pixels.split())