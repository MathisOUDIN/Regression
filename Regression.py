
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.model_selection import train_test_split

house_data = pd.read_csv('house_data.csv')

plt.plot(house_data['surface'], house_data['price'], 'ro', markersize=4)
plt.show()


# In[2]:


plt.plot(house_data['arrondissement'], house_data['price'], 'ro', markersize=4)
plt.show()

#~ price,surface,arrondissement


# In[3]:


house_data[house_data['arrondissement'] == 1]
plt.plot(house_data[house_data['arrondissement'] == 1]['surface'], 
         house_data[house_data['arrondissement'] == 1]['price'], 'ro', markersize=4)
plt.show()


# In[4]:


plt.plot(house_data[house_data['arrondissement'] == 1]['surface'], 
         house_data[house_data['arrondissement'] == 1]['price'], 'ro',
         house_data[house_data['arrondissement'] == 2]['surface'], 
         house_data[house_data['arrondissement'] == 2]['price'], 'bo',
         house_data[house_data['arrondissement'] == 3]['surface'], 
         house_data[house_data['arrondissement'] == 3]['price'], 'go',
         house_data[house_data['arrondissement'] == 4]['surface'], 
         house_data[house_data['arrondissement'] == 4]['price'], 'yo',
         house_data[house_data['arrondissement'] == 10]['surface'], 
         house_data[house_data['arrondissement'] == 10]['price'], 'mo', markersize=4)
plt.show()


# In[5]:


data = house_data[house_data['arrondissement'] > 0]


# In[6]:


X = np.matrix([np.ones(data.shape[0]), data['arrondissement'].as_matrix(), 
               data['surface'].as_matrix()]).T
y = np.matrix(data['price']).T


# In[7]:


theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
print(theta)


# In[8]:


theta.item(0) + theta.item(1) * 1 + theta.item(2) * 35


# In[9]:


theta.item(0) + theta.item(1) * 2 + theta.item(2) * 35


# In[10]:


theta.item(0) + theta.item(1) * 3 + theta.item(2) * 35


# In[11]:


theta.item(0) + theta.item(1) * 4 + theta.item(2) * 35


# In[12]:


theta.item(0) + theta.item(1) * 10 + theta.item(2) * 35


# In[22]:


plt.xlabel('Surface')
plt.ylabel('Loyer')

arrondissement = 1
plt.plot(house_data[house_data['arrondissement'] == arrondissement]['surface'], 
         house_data[house_data['arrondissement'] == arrondissement]['price'], 'ro', markersize=4)
plt.plot([0,250], [theta.item(0),theta.item(0) + theta.item(1) * arrondissement + 250 * theta.item(2)], linestyle='--', c='#000000')


# In[16]:


theta.item(0) + theta.item(1) * 10 + theta.item(2) * 175

