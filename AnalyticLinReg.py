import statistics

'''
This class is used to fit an analytic linear regression model
'''
class AnalyticLinReg:
    """
    This is the constructor for the class AnalyticLinReg
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.theta1 = 0
        self.theta0 = 0

    '''
    This function fits the model to the given data
    The fit function is like a learning function. 
    It takes in the "question" X and look at the "answer" y.
    and then it learns to do prediction once the model 
    encounter the X_pred and y_pred.

    :param X: input values
    :param y: output values
    :return: None
    '''

    def fit(self):
        self.mean_X = statistics.mean(self.X)
        self.mean_y = statistics.mean(self.y)
        numer = 0
        denom = 0
        for i in range(len(self.X)):
            numer += (self.X[i] - self.mean_X) * (self.y[i] - self.mean_y)
            denom += ((self.X[i] - self.mean_X) ** 2)
        if denom == 0:
            raise ValueError("Denominator is zero")
        self.theta1 = numer / denom
        self.theta0 = self.mean_y - self.theta1 * self.mean_X

    '''
    This function calculates the hypothesis for the given input
    :param x: input value
    :return: hypothesis value
    '''
    def hypothesis(self, x):
        return self.theta0 + self.theta1 * x

    '''
    This function predicts the value for the given input
    :param X: input values
    :param y: output values
    :param X_pred: input values for which the output is to be predicted
    :param y_pred: output values for which the output is to be predicted
    :return: y_pred
    '''
    def predict(self, X, y, X_pred, y_pred):
        for i in range(len(X_pred)):
            y_pred[i] = self.hypothesis(X_pred[i])

        return y_pred
