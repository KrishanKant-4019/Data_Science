class LabelEncoder:
    def __init__(self):
        self.classes_ = None  # Dictionary to store label -> number mapping

    def fit(self, y):
        self.classes_ = {label: idx for idx, label in enumerate(np.unique(y))}  # Create mapping

    def transform(self, y):
        if self.classes_ is None:
            raise Exception("You need to fit the encoder first!") 
        return np.array([self.classes_[label] for label in y])  # Convert labels to numbers

    def fit_transform(self, y):
        self.fit(y)           # Fit mapping
        return self.transform(y)  # Transform categories