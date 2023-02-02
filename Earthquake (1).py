#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/database.csv")
data.columns


# In[ ]:


data = data[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]
data.head()


# In[ ]:


import datetime
import time

timestamp = []
for d, t in zip(data['Date'], data['Time']):
    try:
        ts = datetime.datetime.strptime(d+' '+t, '%m/%d/%Y %H:%M:%S')
        timestamp.append(time.mktime(ts.timetuple()))
    except ValueError:
        # print('ValueError')
        timestamp.append('ValueError')
timeStamp = pd.Series(timestamp)
data['Timestamp'] = timeStamp.values
final_data = data.drop(['Date', 'Time'], axis=1)
final_data = final_data[final_data.Timestamp != 'ValueError']
final_data.head()


# In[ ]:


from mpl_toolkits.basemap import Basemap

m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')

longitudes = data["Longitude"].tolist()
latitudes = data["Latitude"].tolist()
#m = Basemap(width=12000000,height=9000000,projection='lcc',
            #resolution=None,lat_1=80.,lat_2=55,lat_0=80,lon_0=-107.)
x,y = m(longitudes,latitudes)

fig = plt.figure(figsize=(12,10))
plt.title("All affected areas")
m.plot(x, y, "o", markersize = 2, color = 'blue')
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
m.drawmapboundary()
m.drawcountries()
plt.show()


# In[ ]:


get_ipython().system('pip install basemap')


# In[ ]:


get_ipython().system('pip install basemap-data')


# In[ ]:


final_data.shape


# In[ ]:


X = final_data[['Timestamp', 'Latitude', 'Longitude']]
y = final_data[['Magnitude', 'Depth']]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[ ]:


get_ipython().system('python -m pip install --upgrade pip')


# In[ ]:


from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


# Create the SVR regressor
rf = RandomForestRegressor()


# In[ ]:


reg = MultiOutputRegressor(rf)


# In[ ]:


model = reg.fit(X_train,y_train)


# In[ ]:


model.score(X_train,y_train)


# In[ ]:


from sklearn.metrics import mean_squared_error,r2_score


# In[ ]:


pred = model.predict(X_test)


# In[ ]:


mean_squared_error(pred,y_test)


# In[ ]:


r2_score(pred,y_test)


# In[ ]:


pred


# In[ ]:


pd.DataFrame(pred)


# In[ ]:




