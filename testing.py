import matplotlib
import pandas as pd
import numpy as np
import sys
sys.path.append('C:\\Users\\yongr\\PycharmProjects\\pythonProject1')
import matplotlib.pyplot as plt

path = r"C:\Users\yongr\PycharmProjects\pythonProject1\boston.csv"

df = pd.read_csv(path)

print(df)

df.head()

df.shape

df.info()

#remove duplicates
df = df.drop_duplicates()
df = df.dropna()

df.describe()

df.columns

# For distribution of numerical features
plt.figure(figsize=(10, 6))
plt.hist(df['CRIM'], bins=30, color='c')
plt.title('CRIM')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df['ZN'], bins=30, color='c')
plt.title('ZN')
plt.show()
# Add more features if needed

plt.scatter(df['RM'],df['MEDV'])
from sklearn.model_selection import train_test_split
# Define features and target variable
X = df.drop('MEDV', axis=1)  # Features
y = df["MEDV"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=101)

if X.ndim == 1:
    X = X[:, None]

if y.ndim == 1:
    y = y[:, None]