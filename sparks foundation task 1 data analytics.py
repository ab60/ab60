#!/usr/bin/env python
# coding: utf-8

# In[8]:


#importing all libraries requred in this notebook
import numpy as np
import pandas as pd
import matplotlip as plt
import seaborn as sns


# In[ ]:


# reading data from remote link
df = pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')


# In[13]:


df.head()


# In[14]:


df.plot(x='Hours', y='Scores', style='.', color='blue')
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.grid()
plt.show()


# In[15]:


# we can also use .corr to determine the corelation between the variables 
df.corr()


# In[16]:


df.head()


# In[17]:



# using iloc function we will divide the data 
X = df.iloc[:, :1].values  
y = df.iloc[:, 1:].values


# In[25]:


X


# In[26]:


y


# In[27]:


# Splitting data into training and testing data

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)


# In[28]:


from sklearn.linear_model import LinearRegression  

l = LinearRegression()  
l.fit(X_train, y_train)


# In[29]:


#To print coefficient and intercepts model
l.coef_


# In[30]:


l.intercept_


# In[31]:


# Plotting the regression line
line = l.coef_*X+l.intercept_

# Plotting for the test data
plt.show()
plt.scatter(X_train, y_train, color='red')
plt.plot(X, line, color='green');
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score') 
plt.grid()
plt.show()


# In[32]:


print(X_test)
y_pred = l.predict(X_test) # Predicting the scores


# In[33]:


#Comparing actual and predicated values of a dataframes
comp = pd.DataFrame({ 'Actual':[y_test],'Predicted':[y_pred] })
comp


# In[34]:


# You can also test with your own data
hours = 9.25
own_pred = l.predict([[hours]])
print("Number of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# In[35]:


from sklearn import metrics  

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:




