#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd


# In[52]:


data = pd.read_csv('2018-E-c-En-train.txt', sep="\t", error_bad_lines=False)


# In[53]:


emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
count = 0
dataframes = []
for em in emotions:
    c = pd.DataFrame(columns = ["Tweet", "emotion"])
    df = data[data[em] == 1]
    df = df[["Tweet", em]]
    df = df.rename(columns={em: "emotion"})
    for index_label, row_series in df.iterrows():
        # For each row update the 'Bonus' value to it's double
        df.at[index_label , 'emotion'] = count
        c = c.append({"Tweet":df.at[index_label , 'Tweet'], 
                      "emotion": df.at[index_label , 'emotion']}, ignore_index=True)
        df.drop(index_label, inplace=True)        
    count += 1
    dataframes.append(c)


# In[54]:


#now it is time to take care of the neutral
data = pd.read_csv("text_emotion.csv")
data = data.rename(columns={"sentiment" : "emotion", "content" : "Tweet"})
data = data[["Tweet", "emotion"]]
data = data[data["emotion"] == 'neutral']
for index_label, row_series in data.iterrows():
    if(data.at[index_label, 'emotion'] == "neutral"):
        data.at[index_label , 'emotion'] = 6
dataframes.append(data)


# In[55]:


main = pd.concat(dataframes).reset_index(drop=True)    
main["emotion"].unique()


# In[56]:


import nltk
nltk.download('punkt')
import re
import string
punctuation = string.punctuation
from nltk.corpus import stopwords
stopwords = stopwords.words("english")


# In[58]:


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


# In[59]:


main["clean tweets"] = main["Tweet"].apply(cleanTweet)
main.to_csv("tweets.csv")
main


# In[ ]:




