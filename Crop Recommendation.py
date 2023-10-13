#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("C:\\Users\\Sricharan Reddy\\Downloads\\flask_int_roh\\Crop_recommendation.csv")


# In[3]:


df.head()


# In[59]:


df['label'].unique()


# In[4]:


df.info()


# In[53]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[56]:


df


# In[54]:


x = df.drop('label',axis = 1)


# In[ ]:





# In[55]:


x.head()


# In[7]:


y = pd.get_dummies(df['label'] , drop_first = True)


# In[57]:


y


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2 , random_state = 9)


# In[11]:


from xgboost import XGBClassifier


# In[12]:


x_model = XGBClassifier()


# In[14]:


x_model.fit(x_train, y_train)


# In[25]:


y_pred_train = x_model.predict(x_train)


# In[26]:


y_pred_test = x_model.predict(x_test)


# In[27]:


from sklearn.metrics import accuracy_score


# In[28]:


accuracy_score(y_train,y_pred_train)


# In[29]:


accuracy_score(y_test,y_pred_test)


# In[43]:


query = np.array([[90,42,43,20.8,82,6.5,202]])


# In[61]:


output_array = x_model.predict(query)


# In[64]:


categories = ['banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute',
              'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
              'pigeonpeas', 'pomegranate', 'rice', 'watermelon']


# In[66]:


category_index = np.argmax(output_array)

predicted_category = categories[category_index]

print("Predicted Category:", predicted_category)


# In[67]:


query = np.array([[88,38,15,25,55,5.6,78]])


# In[68]:


output_array = x_model.predict(query)
category_index = np.argmax(output_array)

predicted_category = categories[category_index]

print("Predicted Category:", predicted_category)


# In[69]:


import pickle


# In[71]:


pickle.dump(x_model,open('model_x.pkl','wb'))


# In[ ]:




