"""
DataLoader loads the data from fer2013.csv and creates the train and test sets
used by CNN
"""

import pandas as pd
from TaggedImage import *
from sklearn.model_selection import train_test_split


class DataLoader:

    def __init__(self):
        self.dataset = pd.read_csv('fer2013.csv')

        train, test = train_test_split(self.dataset, test_size=0.3)
        self.train = self.process_data(train)
        self.test = self.process_data(test)

    def process_data(self, data):
        tagged_image_list = []
        for csv_row in data.iterrows():
            tagged_image_list.append(TaggedImage(csv_row[1]))
        return tagged_image_list


# load = DataLoader()
# print load.train[0].pixels
# print load.train[1].label
