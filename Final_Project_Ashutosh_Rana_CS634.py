#!/usr/bin/env python
# coding: utf-8

# # Importing Libreries
# 

# In[564]:



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # pip install pandas
import seaborn as sns
from prettytable import PrettyTable
from random import randint


# In[565]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ### Reading the dataset

# In[566]:


df = pd.read_csv('prices.csv')


# #### Droping null values and printing first 5 rows

# In[567]:


df=df.dropna()
df.head()


# In[568]:


#information about the dataset
df.info()


# ### Some stats about the dataset

# In[569]:


df.describe()


# In[570]:


#Column of the dataset
df.columns


# #### Some graphs about all field in relation to other fields

# In[571]:


sns.pairplot(df)


# Heatmap

# In[572]:


sns.heatmap(df.corr(),annot=True)


# In[573]:


#slected any stock for whom you want the precdiction
ticker = df['symbol'].unique() 
t = PrettyTable(['Ticker'])
for i in range(len(ticker)):
    t.add_row([ticker[i]])
print(t)


# In[574]:


selected_symbol = input('Please enter the ticker from table above: ').upper()
while selected_symbol not in ticker:
    selected_symbol = input('Please enter the valid ticker from above table: ').upper()
print(f"{selected_symbol} is selected.")


# Selecting rows for the stock ticker selected
# 

# In[575]:


df = df.loc[df['symbol']== selected_symbol]
df.drop(['symbol','open','low','high','volume'],inplace=True,axis=1)


# In[576]:


df.head()


# ##### Selcting the closing price of the selected stock

# In[577]:


x_data = df['close']


# #### Adding noise to the data

# In[578]:


noise = np.random.randn(len(x_data))


# #### Cost Function
# ######  y = mx + b

# In[606]:


b = 5
m =0.5
y_true = (0.5*x_data)+5+noise #original y


# ##### Concatinating X data and y_true

# In[607]:


df1 = pd.DataFrame({'X Data': x_data})
df2 = pd.DataFrame({'Y': y_true})
my_data=pd.concat([df1, df2],axis = 1)
my_data.head()


# ##### Ploting scatter graph of x_data vs original cost function

# In[608]:


my_data.sample(n=250).plot(kind='scatter',x='X Data',y='Y')


# # Tensorflow

# In[582]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# ##### Batch size

# In[583]:


# Random 10 points to grab
batch_size = 8


# ** Variables **

# In[584]:


m = tf.Variable(0.5)
b = tf.Variable(1.0)


# ** Placeholders **

# In[585]:


xph = tf.placeholder(tf.float32,[batch_size])


# In[586]:


yph = tf.placeholder(tf.float32,[batch_size])


# ** Our graph **

# In[609]:


y_model = m*xph + b # y for trainging the model


# ** Loss Function **

# In[588]:


error = tf.reduce_mean(tf.square(yph-y_model))


# ** Optimizer **

# In[601]:


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
train = optimizer.minimize(error)


# ** Initial Variables **
# 

# In[590]:


init = tf.global_variables_initializer()


# ** Array of indices of x_data **

# In[602]:


x_data_index = x_data.index


# ## Session

# In[592]:



with tf.Session() as sess:
    sess.run(init)
    batches = 100
    for i in range(batches):
        rand_ind = np.random.randint(len(x_data_index),size=batch_size)
        feed = {xph:x_data[x_data_index[rand_ind]],yph:y_true[x_data_index[rand_ind]]}
        sess.run(train,feed_dict=feed)
    model_m,model_b = sess.run([m,b])


# In[593]:


model_m


# In[594]:


model_b


# ### Results

# In[610]:


y_hat = x_data * model_m + model_b #predicted Y


# In[612]:


my_data.sample(n=500).plot(kind='scatter',x='X Data',y='Y')
plt.plot(x_data,y_hat,'r')


# In[613]:


df1 = pd.DataFrame({'Y Original': y_true})
df2 = pd.DataFrame({'Y Predicted': y_hat})
my_data=pd.concat([df1, df2],axis = 1)
my_data


# In[ ]:




