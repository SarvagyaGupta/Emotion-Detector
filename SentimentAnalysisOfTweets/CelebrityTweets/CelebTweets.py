#!/usr/bin/env python
# coding: utf-8

# In[164]:


import numpy as np
import pandas as pd
import re
import csv
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_similarity_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import regularizers, initializers, optimizers, callbacks
import nltk
nltk.download('punkt')
import re
import string
punctuation = string.punctuation
from nltk.corpus import stopwords
stopwords = stopwords.words("english")


# In[165]:


celeb = pd.read_csv("celeb.csv")


# In[166]:


def cleanTweet(tweet):
    tweet = tweet.lower()
    #maybe we want to remove the twitter handle as well
    #removing the punctuations
    tweet = "".join(x for x in tweet if x not in punctuation)
    #removing the stop words
    individualWords = tweet.split()
    individualWords = [w for w in individualWords if w not in stopwords]
    tweet = " ".join(individualWords)
    #removing certain common occuring words
    tweet = re.sub('rt', '', tweet)
    return tweet


# In[167]:


celeb = celeb[celeb["language"] == 'en']
celeb = celeb[["author", "content"]]
celeb["clean tweets"] = celeb["content"].apply(cleanTweet)
celeb = celeb.sample(frac=1).reset_index(drop=True)
celeb = celeb[:6000]
celeb.to_csv("CelebrityTweets.csv")


# In[10]:


from keras.models import model_from_json
# Load the model
emotion_file = open('twitter.json', 'r')
loaded_model_json = emotion_file.read()
emotion_file.close()
modeled_emotion = model_from_json(loaded_model_json)

# Load the weights
modeled_emotion.load_weights('twitter.h5')


# In[168]:


tweets = celeb["clean tweets"]
t = []
for i in tweets:
    t.append(i)
t


# In[17]:


MAX_NB_WORDS = 40000
MAX_SEQUENCE_LENGTH = 30
VALIDATION_SPLIT = 0.2


# In[141]:



tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(t)
sequences = tokenizer.texts_to_sequences(t)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)


# In[169]:


probs = []
# for x in data:
#     print(x.shape)
z = 1
r = 0
while z != len(t) + 1:
        y = modeled_emotion.predict(data[r:z])
        probs.append(y)
        r += 1
        z += 1


# In[170]:


len(probs)


# In[147]:


len(t)


# In[171]:


celeb["probs"] = probs


# In[154]:


# celeb['probs'][0][0] for accessing the probabilities


# In[175]:


celeb.to_csv("FinalCelebrityTweets.csv")


# In[173]:





# In[ ]:




