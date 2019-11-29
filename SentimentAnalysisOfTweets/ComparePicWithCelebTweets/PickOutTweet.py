#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import sys
import math


# In[31]:


probability = [0.000509, 0.0, 0.262479, 8.3e-05, 7.4e-05, 0.002448, 0.734407]
emotion = "Neutral"


# In[45]:


data = pd.read_csv("FinalCelebrityTweets.csv")


# In[46]:


minimum = sys.maxsize


# In[47]:


minimum


# In[48]:


def calculateCloseness(x, y):
    total = 0
    for i in range(len(x)):
        total += (x[i] - y[i]) ** 2
    total = math.sqrt(total)
    return total


# In[49]:


tweet = ''
person = ''
pr = []
for index_label, row_series in data.iterrows():
    temp = [float(x) for x in data.at[index_label, 'probs'][2:-2].split()]
    p = calculateCloseness(temp, probability)
    if p < minimum:
        minimum = p
        tweet = data.at[index_label, 'content']
        person = data.at[index_label, 'author']
        pr = temp


# In[50]:


tweet


# In[51]:


person


# In[52]:


pr


# In[ ]:




