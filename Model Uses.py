#!/usr/bin/env python
# coding: utf-8

# In[1]:


from joblib import dump, load
import numpy as np
model = load('RealEstate.joblib')


# In[4]:


features = np.array([[-0.44241248,  3.18716752, -1.12581552, -0.27288841, -1.42038605,
      -0.55121782, -1.7412613 ,  2.56284386, -0.99534776, -0.57387797,
      -0.99428207,  0.43852974, -0.49833679]])
model.predict(features)


# In[ ]:




