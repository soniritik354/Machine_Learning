#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the required libraries
import pandas as pd
import numpy as np


# In[2]:


#Visualisation Libraries
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


#Reading the Data
df = pd.read_csv("USA_Housing.csv")
df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[8]:


df.columns


# In[9]:


sns.pairplot(df)


# In[11]:


sns.distplot(df['Price'],bins = 10)


# In[14]:


sns.heatmap(df.corr(),cmap='coolwarm',annot = True)


# In[15]:


df.columns


# In[24]:


X = df.drop(["Price","Address"],axis = 1)
y = df["Price"]


# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.4, random_state=101)


# In[27]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[28]:


lr.fit(X_train,y_train)


# In[29]:


lr.intercept_


# In[30]:


lr.coef_


# In[31]:


X.columns


# In[33]:


cdf = pd.DataFrame(lr.coef_,X.columns,columns=["Coeff"])
cdf


# In[36]:


Predictions = lr.predict(X_test)
Predictions


# In[39]:


plt.scatter(y_test,Predictions)


# In[41]:


sns.distplot((y_test-Predictions))


# In[44]:


#Evaluation Metrics for Regression
from sklearn import metrics
print(f"MAE : {metrics.mean_absolute_error(y_test,Predictions)}")
print(f"MSE : {metrics.mean_squared_error(y_test,Predictions)}")
print(f"RMSE : {np.sqrt(metrics.mean_squared_error(y_test,Predictions))}")

