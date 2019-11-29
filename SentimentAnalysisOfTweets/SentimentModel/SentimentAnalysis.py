#!/usr/bin/env python
# coding: utf-8

# In[51]:


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


# In[75]:


labels = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
data = pd.read_csv("tweets.csv")


# In[76]:


# remove the unnamed column
data = data[["Tweet", "clean tweets", "emotion"]]
data.to_csv("finaltweet.csv")


# In[77]:


texts, labels, clean = [], [], []
with open('finaltweet.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        texts.append(row[1])
        labels.append(row[3])
        clean.append(row[2])


# In[78]:


labels = labels[1:]
clean = clean[1:]


# In[116]:


clean


# In[80]:


MAX_NB_WORDS = 40000
MAX_SEQUENCE_LENGTH = 30
VALIDATION_SPLIT = 0.2


# In[81]:


tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(clean)
sequences = tokenizer.texts_to_sequences(clean)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)


# In[117]:


data


# In[82]:


from sklearn.model_selection import train_test_split


# In[83]:


from keras.utils.np_utils import to_categorical
labels = to_categorical(np.asarray(labels))
labels


# In[84]:


Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, labels , test_size=0.3)


# In[42]:


EMBEDDING_DIM = 100
GLOVE_DIR = "glove.twitter.27B.100d.txt"
embeddings_index = {}
f = open(GLOVE_DIR)
print("Loading GloVe from:",GLOVE_DIR,"...",end="")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print("Done.\nProceeding with Embedding Matrix...")
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
print("Completed!")


# In[64]:


embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


# In[65]:


sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32') # 1000
embedded_sequences = embedding_layer(sequence_input)
l_cov1= Conv1D(128, 3, activation='relu',kernel_regularizer=regularizers.l2(0.01))(embedded_sequences)
l_cov2= Conv1D(128, 3, activation='relu',kernel_regularizer=regularizers.l2(0.01))(l_cov1)
l_pool1 = MaxPooling1D(3)(l_cov2)
#l_drop1 = Dropout(0.2)(l_pool1)
l_cov3 = Conv1D(128, 3, activation='relu',kernel_regularizer=regularizers.l2(0.01))(l_pool1)
l_pool4 = MaxPooling1D(6)(l_cov3) # global max pooling
l_flat = Flatten()(l_pool4)
l_dense = Dense(128, activation='relu')(l_flat)
preds = Dense(7, activation='softmax')(l_dense) # 4 categories


# In[67]:


def step_cyclic(epoch):
    try:
        if epoch%11==0:
            return float(2)
        elif epoch%3==0:
            return float(1.1)
        elif epoch%7==0:
            return float(0.6)
        else:
            return float(1.0)
    except:
        return float(1.0)
        
tensorboard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=256, write_grads=True , write_graph=True)
model_checkpoints = callbacks.ModelCheckpoint("checkpoints", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
lr_schedule = callbacks.LearningRateScheduler(step_cyclic)


# In[113]:


import pandas as pd
tweets = [
    "Watching the sopranos again from start to finish!",
    "Finding out i have to go to the  dentist tomorrow",
    "I want to go outside and chalk but I have no chalk",
    "I HATE PAPERS AH #AH #HATE",
    "My mom wasn't mad",
    "Do people have no Respect for themselves or you know others peoples homes",
]

def convert(texts):
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return data


# In[86]:


k = Model(sequence_input, preds)
adadelta = optimizers.Adadelta(lr=0.01, rho=0.95, epsilon=None, decay=0.01)
k.compile(loss='categorical_crossentropy',
              optimizer=adadelta,
              metrics=['acc'])
k.summary()


# In[87]:


k.fit(Xtrain, Ytrain,
          epochs=200, batch_size=128,
          callbacks=[tensorboard, model_checkpoints, lr_schedule])


# In[127]:


y_hat = k.predict(Xtest)
count = 0
for i in range(len(y_hat)):
    if np.argmax(y_hat[i]) == np.argmax(Ytest[i]):
        count += 1

print(100 * count / len(y_hat))


# In[95]:


saved_model = k.to_json()
with open("twitter.json", "w") as json_file:
    json_file.write(saved_model)
model.save_weights("twitter.h5")


# In[120]:


y_hat = k.predict(tweets)
for i in range(len(y_hat)):
    print(np.argmax(y_hat[i])) 


# In[ ]:




