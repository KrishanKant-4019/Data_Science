# Importing numpy library
import numpy as np

# Logistic Regression
class Logistic_Regression():

    # Declaring learning rate & number of iterations(hyperparameters)
    def __init__(self, learning_rate, no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    # Fit function to train the model with dataset
    def fit(self, X, Y):

        # Number of data points in the dataset(number of rows) --> m
        # Number of input features in the dataset(number of columns) --> n 
        self.m, self.n = X.shape

        # Initiating weight & bias values
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        # Implementing Gradient Descent for Optimization
        for i in range(self.no_of_iterations):
            self.update_weights()
    
    def update_weights(self):

        # Sigmoid Function(Y_hat)
        Y_hat = 1 / (1 + np.exp( - (self.X.dot(self.w) + self.b) )) # wX + b

        # Derivatives
        dw = (1 / self.m) * np.dot(self.X.T, (Y_hat - self.Y))
        db = (1 / self.m) * np.sum(Y_hat - self.Y) 

        # Updating the weights & bias using Gradient Descent
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    # Sigmoid Function & Decision Boundary
    def predict(self, X):
        Y_pred = 1 / (1 + np.exp( - (X.dot(self.w) + self.b) ))
        Y_pred = np.where(Y_pred > 0.5, 1, 0)
        return Y_pred