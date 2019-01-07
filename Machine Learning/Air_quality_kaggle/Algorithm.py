#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[16]:


data = pd.read_csv("Train.csv")


# In[17]:


data.head(5)


# In[18]:


data = data.values


# In[28]:


print (data[:,0])


# In[59]:


def hypothesis(X,theta):
    return theta[0]+theta[1]*X[0]+theta[2]*X[1]+theta[3]*X[2]+theta[4]*X[3]+theta[5]*X[4]


# In[30]:


X=data[:,:5]


# In[31]:


X.shape


# In[33]:


type(X)


# In[34]:


ones = np.ones((X.shape[0],1))


# In[35]:


X=np.hstack((ones,X))
X.shape


# In[36]:


print(X[:5,:])


# In[37]:


data[:,5]


# In[38]:


Y=data[:,5]


# In[39]:


Y.shape


# In[42]:


Y= Y.reshape((-1,1))


# In[43]:


Y.shape


# In[44]:


# Applying Formula For Closed Form


# In[45]:


theta = np.dot(np.linalg.pinv(np.dot(X.T,X)),np.dot(X.T,Y))


# In[46]:


theta.shape


# In[47]:


theta = theta.reshape((-1,))


# In[48]:


print(theta.shape)
print(theta)


# # Testing The Algorithm

# In[61]:


test = pd.read_csv("Test.csv")


# In[62]:


test.head(5)


# In[63]:


pred = []
test.shape
test = test.values
test.shape


# In[66]:


test[0,:]


# In[67]:


for ix in range(test.shape[0]):
    pred.append(hypothesis(test[ix,:],theta))


# In[77]:


pred = np.array(pred)


# In[79]:


pred.shape


# In[81]:


pred = pred.reshape(-1,1)


# In[82]:


pred.shape


# In[103]:


#fin=[]


# In[104]:


#for ix in range(pred.shape[0]):
 #   fin.append([ix,pred[ix]])


# In[105]:


#fin = np.array(fin)


# In[106]:


#fin.shape


# In[ ]:





# In[116]:


sub = pd.DataFrame(pred)


# In[117]:


sub.to_csv("Submission.csv")


# In[125]:


tsub = pd.read_csv("Submission.csv")


# In[126]:


tsub.head(5)


# In[127]:


tsub.columns


# In[128]:


tsub.columns = ['Id','target']


# In[129]:


tsub.columns


# In[130]:


tsub.head(5)


# In[137]:


tsub.to_csv("Submission1.csv",index = False)


# In[138]:


msub = pd.read_csv("Submission1.csv")


# In[139]:


msub.head(5)


# In[135]:


msub.drop('Unnamed: 0', axis=1, inplace=True)


# In[136]:


msub.head(5)


# In[ ]:




