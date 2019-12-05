"""
DataLoader loads the data from fer2013.csv and creates the train and test sets
used by CNN
"""

import pandas as pd
import numpy as np
from TaggedImage import TaggedImage
from sklearn.model_selection import train_test_split


class DataLoader:

    def __init__(self, num_emotions):
        self.dataset = pd.read_csv('../fer2013.csv')

        self.num_emotions = num_emotions
        self.train, test = train_test_split(self.dataset, test_size=0.3)
        validation, test = train_test_split(test, test_size=0.3)
        self.__data_augment()
        self.train = self.__process_data(self.train)
        self.validation = self.__process_data(validation)
        self.test = self.__process_data(test)

    def __process_data(self, data):
        tagged_image_list = []
        for csv_row in data.iterrows():
            tagged_image_list.append(TaggedImage(csv_row[1], self.num_emotions))
        return tagged_image_list

    def __data_augment(self):
        count = len(self.train)
        curr_count = 0
        for csv_row in self.train.iterrows():
            if curr_count == count:
                break
            curr_count += 1
            im = csv_row[1]['pixels'].split()
            im = np.reshape(im, (48,48))
            em = csv_row[1]['emotion']
            p = np.fliplr(im)
            p = p.flatten()
            p = " ".join(p)
            self.train = self.train.append({'pixels': p, 'emotion': em, 'Usage': "Training"}, ignore_index=True)
