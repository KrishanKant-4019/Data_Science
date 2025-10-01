import numpy as np

class MinMaxScaler:
    def __init__(self):
        self.min_ = None  # to store column-wise minimum value
        self.max_ = None  # to store column-wise maximum value

    # Step 1: Learn min and max from training data
    def fit(self, X):
        self.min_ = np.min(X, axis=0)   # column-wise min
        self.max_ = np.max(X, axis=0)   # column-wise max

    # Step 2: Transform data to [0,1] range using learned minimum and maximum value
    def transform(self, X):
        if self.min_ is None or self.max_ is None:
            raise Exception("You need to fit the scaler first!")
        return (X - self.min_) / (self.max_ - self.min_)

    # Step 3: Fit and transform together
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)