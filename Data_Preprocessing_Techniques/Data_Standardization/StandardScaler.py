import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None  # to store column-wise mean
        self.std_ = None   # to store column-wise standard deviation

    # Step-1: Learn the mean and std from training data
    def fit(self, X):
        self.mean_ = np.mean(X, axis = 0)  # calculate mean of each feature
        self.std_ = np.std(X, axis = 0)  # calculate standard deviation of each feature

    # Step-2: Transform data using learned mean and std
    def transform(self, X):
        if self.mean_ is None or self.std_ is None:
            raise Exception("You need to fit the scaler first!")
        return (X - self.mean_) / self.std_

    # Step-3: Fit and transform at once
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)