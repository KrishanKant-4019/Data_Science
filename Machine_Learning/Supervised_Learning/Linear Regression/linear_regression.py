# Importing numpy library
import numpy as np

class Linear_Regression:

    # Initiating the parameters (learning rate & no. of iterations)
    def __init__(self, learning_rate, no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    def fit(self, X, Y):
        self.m, self.n = X.shape # Number of rows & columns

        # Initiating the weight and bias
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        # Implementing Gradient Descent
        for i in range(self.no_of_iterations):
            self.update_weights()  # Fixed typo from 'udpate_weights'
        
    def update_weights(self):
        Y_prediction = self.predict(self.X)

        # Calculating gradients
        dw = -(2 * (self.X.T).dot(self.Y - Y_prediction)) / self.m
        db = -2 * np.sum(self.Y - Y_prediction) / self.m

        # Updating the weights
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db
        
    def predict(self, X):
        return X.dot(self.w) + self.b
