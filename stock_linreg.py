# imports the necessary libraries for data manipulation 
# and visualization (numpy, pandas, and matplotlib), and 
# machine learning (sklearn). 

import numpy as np 
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

# reads and saves it to a pandas dataframe 'df' 
df=pd.read_csv(r'TSLA.csv')

# print(list[df.columns])
x=df[['Open','High','Low','Volume']].to_numpy()
y=df.Close.to_numpy()

# creates an instance of the Logistic Regression model 
# and assigns it to the variable 'model'
model = LinearRegression()

# 'model' is fit to the data 'x' and 'y' using the fit() method
model.fit(x,y)

# the score of the model is printed using the score() method which 
# returns the mean accuracy on the given test data and labels.
print(model.score(x,y))


# the predict() method to predict the probability of the given sample height=180cm

print("prediction for close(343.503326)  ",model.predict([[342.203339,356.929993,338.686676,66743400]]))

df['Date']=pd.to_datetime(df['Date'])
# df.set_index('Date',inplace=True)

plt.title(' Stock Predition graph for TESLA ')
plt.scatter(df.Date,df['Adj Close'])
plt.xlabel("  Date  ")
plt.ylabel("  Adj Close  ")
plt.plot(df.Date,df['Adj Close'],color='r')
plt.show()

# then uses the pickle library to save the model to a file named "stock-linreg-model.pkl" and 
# later on it loads the model from the same file using the load() method of pickle
pickle.dump(model, open("stock-linreg-model.pkl", "wb"))
model=pickle.load(open('stock-linreg-model.pkl','rb'))
