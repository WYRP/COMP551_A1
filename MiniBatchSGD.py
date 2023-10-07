import matplotlib
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

path = r"C:\Users\yongr\PycharmProjects\pythonProject1\boston.csv"

df = pd.read_csv(path)

print(df)

df.head()

df.shape

df.info()

# remove duplicates
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

plt.scatter(df['RM'], df['MEDV'])
from sklearn.model_selection import train_test_split

# Define features and target variable
X = df.drop('MEDV', axis=1)  # Features
y = df["MEDV"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

np.array(y)
y = np.array(y)
if y.ndim == 1:
    y = y[:,None]

print(len(y.shape))

# linear regression# linear regression using "mini-batch" gradient descent
# function to compute hypothesis / predictions


# function to compute gradient of error function w.r.t. theta
def hypothesis(X, theta):
	return np.dot(X, theta)

def gradient(X, y, theta):
    h = hypothesis(X, theta)
    grad = np.dot(X.transpose(), (h - y))
    return grad


# function to compute the error for current values of theta


def cost(X, y, theta):
    h = hypothesis(X, theta)
    J = np.dot((h - y).transpose(), (h - y))
    J /= 2
    return J[0]


# function to create a list containing mini-batches


def create_mini_batches(X, y, batch_size):
    mini_batches = []
    print(X.shape)
    y = np.array(y)
    if y.ndim == 1:
        y = y[:,None]
    print(y.shape)
    data = np.hstack((X, y))
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size
    i = 0

    for i in range(n_minibatches + 1):
        mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    if data.shape[0] % batch_size != 0:
        mini_batch = data[i * batch_size:data.shape[0]]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    return mini_batches


# function to perform mini-batch gradient descent


def gradientDescent(X, y, learning_rate=0.0001, batch_size=8):
    theta = np.zeros((X.shape[1], 1))
    error_list = []
    max_iters = 3
    for itr in range(max_iters):
        mini_batches = create_mini_batches(X, y, batch_size)
        for mini_batch in mini_batches:
            X_mini, y_mini = mini_batch
            theta = theta - learning_rate * gradient(X_mini, y_mini, theta)
            error_list.append(cost(X_mini, y_mini, theta))

    return theta, error_list


# predicting output for X_test

theta, error_list = gradientDescent(X_train, y_train)

y_pred = hypothesis(X_test, theta)
plt.scatter(X_test[:, 1], y_test[:, ], marker='.')
plt.plot(X_test[:, 1], y_pred, color='orange')
plt.show()

print("Bias = ", theta[0])
print("Coefficients = ", theta[1:])

# visualising gradient descent
plt.plot(error_list)
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.show()
