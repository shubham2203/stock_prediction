#!/usr/bin/env python
# coding: utf-8

# In[77]:


import numpy as np
import pandas as pd


# In[78]:


df=pd.read_excel('/Users/Shubham Shah/Desktop/Jupyter Notebook/Stock Market WebApp/Stock Market/LSTM/icici.xlsx')


# In[79]:


df.head()


# In[80]:


X = df.iloc[:,[2,3,4,5,6,7,8,9,10,11,12,13]].values
Y = df.iloc[:,[1]].values


# In[81]:


X.shape


# In[82]:


Y


# In[83]:


Y.shape


# In[84]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[85]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
X_new = sc.fit_transform(X)
Y_new=sc.fit_transform(Y)


# In[86]:


X_new


# In[87]:


Y_new


# In[88]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X_new,Y_new,test_size=0.30,random_state=0)


# In[89]:


from sklearn.ensemble import RandomForestRegressor
classifier = RandomForestRegressor(n_estimators=100,random_state=0)
classifier.fit(X_train,Y_train)


# In[ ]:





# In[90]:


y_pred = classifier.predict(X_test)


# In[91]:


y_pred=y_pred.reshape(-1,1)


# In[92]:


y_pred.shape


# In[93]:


y_pred = sc.inverse_transform(y_pred)


# In[94]:


Y_test=sc.inverse_transform(Y_test)
Y_train=sc.inverse_transform(Y_train)


# In[95]:




plt.plot(Y_test)
plt.plot(y_pred)
plt.plot()


# In[96]:


from sklearn.metrics import mean_squared_error, r2_score

print('Mean squared error: %.2f' % mean_squared_error(Y_test, y_pred))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f' % r2_score(Y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:




