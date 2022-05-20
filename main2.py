import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r"C:\Users\DIBYAJYOTI HALOI\OneDrive\Desktop\c++++++\datasheet.csv")
#print(df)
#print(df.keys())
# Index(['Areas', 'Price'], dtype='object')
#print(df.Areas)
#print(df.Price)
df_X = df[['Areas']]
df_Y = df[['Price']]
#print(df_X)
#print(df_Y)
df_X_train = df_X[:20]
#print(df_X_train)
df_X_test = df_X[20:]
#print(df_X_test)
df_Y_train = df_Y[:20]
#print(df_Y_train)
df_Y_test = df_Y[20:]
#print(df_Y_test)

model = linear_model.LinearRegression()
model.fit(df_X_train, df_Y_train)

df_Y_Predicted = model.predict(df_X_test)
#print(df_Y_Predicted)

print("Mean Squared Error is: ", mean_squared_error(df_Y_test, df_Y_Predicted))
print("Weights ", model.coef_)
print("Intercepts ", model.intercept_)

plt.scatter(df_X_test, df_Y_test)
plt.plot(df_X_test, df_Y_Predicted)
plt.show()

#Mean Squared Error is:  2097013.7116575725
#Weights  [[12.47029353]]
#Intercepts  [475565.19223867]
