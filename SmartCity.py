import pandas as pd
import numpy as np
df=pd.DataFrame
df
df=pd.read_csv("C:/Users/Meghana/Documents/Project9_smart-city-traffic-patterns/smart-city-traffic-patterns/smart city.csv")
df.head(10)
df.dtypes
import matplotlib.pyplot as plt
%matplotlib inline
plt.scatter(df['Junction'],df['Vehicles'])
plt.scatter(df['DateTime'],df['Vehicles'])
x=df[['Vehicles','Junction']]
y=df['DateTime']
x
y
x_test
# Importing LabelEncoder from Sklearn 
# library from preprocessing Module.
from sklearn.preprocessing import LabelEncoder

# Creating a instance of label Encoder.
le = LabelEncoder()

# Using .fit_transform function to fit label
# encoder and return encoded label
label = le.fit_transform(df['DateTime'])

# printing label
label
df["DateTime"]=label
df
x
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5)
x_train
x_test
from sklearn.linear_model import LinearRegression
clf=LinearRegression()
y_train
clf.fit(x_train,y_train)
clf.predict(x_test)
clf.score(x_test,y_test)
