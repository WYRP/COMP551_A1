import statistics


class MiniBSGDLinearReg:
    def __init__(self, X, y, learning_rate, batch_size, num_epochs):
        self.X = X
        self.y = y
        self.mean_X = statistics.mean(X)
        self.mean_y = statistics.mean(y)
        self.theta0 = 0
        self.theta1 = 0
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def computeGradient(self, X, y, theta0, theta1):
        gradient_theta0 = 0
        gradient_theta1 = 0

        for i in range(len(X)):
            hypothesis = theta0 + theta1 * X[i]
            error = hypothesis - y[i]
            gradient_theta0 += error
            gradient_theta1 += error * X[i]

        return gradient_theta0 / len(X), gradient_theta1 / len(X)

    def minibatchSGD(self):
        theta0 = 0
        theta1 = 0

        for epoch in range(self.num_epochs):
            self.X, self.y = shuffle(self.X, self.y)

            for batch_start in range(0, len(self.X), self.batch_size):
                if batch_start + self.batch_size > len(self.X):
                    mini_batch_X = self.X[batch_start:]
                    mini_batch_y = self.y[batch_start:]
                else:
                    mini_batch_X = self.X[batch_start: batch_start + self.batch_size]
                    mini_batch_y = self.y[batch_start: batch_start + self.batch_size]

                gradient_theta0, gradient_theta1 = self.computeGradient(mini_batch_X, mini_batch_y, theta0, theta1)
                theta0 -= self.learning_rate * gradient_theta0
                theta1 -= self.learning_rate * gradient_theta1

        return theta0, theta1

    def predict(self, X):
        return self.theta0 + self.theta1 * X

    def fit(self):
        # Initialize model parameters
        initial_theta0 = 0
        initial_theta1 = 0

        # Call minibatchSGD to update the model parameters
        self.theta0, self.theta1 = self.minibatchSGD(initial_theta0, initial_theta1)