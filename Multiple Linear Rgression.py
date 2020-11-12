#!/usr/bin/env python
# coding: utf-8

# # IMPORTING DATA

# In[1]:


import pandas as pd


# In[2]:


adv = pd.read_csv('/Users/neelsheth/Downloads/Advertising.csv')


# In[3]:


# Gives first 5 rows
adv.head()


# In[4]:


# Gives last 5 rows
adv.tail()


# #  DATA VISUALISATION

# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


sns.pairplot(adv)


# In[7]:


# SIZE is used for total size of graph & ASPECT is used for width of graph
sns.pairplot(adv, x_vars = ['TV', 'radio', 'newspaper'], y_vars = 'sales', size = 9, aspect = 0.7, kind = 'scatter')


# # SPLITTING DATA INTO TRAIN & TEST

# In[8]:


# Dividing independent columns into X variable and dependent into Y
X = adv[['TV', 'radio', 'newspaper']]
y = adv[['sales']]


# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 5)


# # PERFORMING LINEAR REGRESSION

# In[10]:


from sklearn.linear_model import LinearRegression


# In[11]:


lr = LinearRegression()


# In[12]:


# Fitting model into dataset
lr.fit(X_train, y_train)


# In[13]:


# Prediction of values
y_pred = lr.predict(X_test)


# # CHECKING ACCURACY & MSE 

# In[14]:


from sklearn.metrics import r2_score, mean_squared_error


# In[15]:


r2 = r2_score(y_test, y_pred)


# In[16]:


mse = mean_squared_error(y_test, y_pred)


# In[17]:


print ('Model accuracy is = ', r2)
print ('Mean Square Error is = ', mse)

