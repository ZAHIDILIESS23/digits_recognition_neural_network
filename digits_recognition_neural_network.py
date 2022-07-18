#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[2]:


(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()


# In[3]:


len(X_train)


# In[4]:


len(X_test)


# In[5]:


X_train[0].shape


# In[6]:


plt.matshow(X_train[0])


# In[7]:


X_train = X_train / 255
X_test = X_test / 255 # to have 0 or 1


# In[8]:


X_train[0]


# In[9]:


X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)


# In[10]:


X_train_flattened.shape


# In[11]:


model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid') # 10 neurones,and 784 input
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',# errors
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)


# In[12]:


model.evaluate(X_test_flattened, y_test)


# In[13]:


y_predicted = model.predict(X_test_flattened)
y_predicted[0]


# ### np.argmax finds a maximum element from an array and returns the index of it

# In[14]:


np.argmax(y_predicted[0])


# In[17]:


np.argmax(y_predicted[0])
y_predicted_labels = [np.argmax(i) for i in y_predicted]


# In[18]:


cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
cm


# In[19]:


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# ### Using hidden layer

# In[20]:


model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)


# In[21]:


model.evaluate(X_test_flattened,y_test)


# In[22]:


y_predicted = model.predict(X_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# ### Using Flatten layer so that we don't have to call .reshape on input dataset

# In[23]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)


# In[24]:


model.evaluate(X_test,y_test)


# In[26]:


y_predicted = model.predict(X_test)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:




