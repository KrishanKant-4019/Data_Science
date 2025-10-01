class OrdinalEncoder:
    def __init__(self):
        self.mapping = None  # Stores category-to-number mapping
    
    def fit(self, categories):
        self.mapping = {cat: idx for idx, cat in enumerate(categories)}  # Create mapping 
    
    def transform(self, X):
        if self.mapping is None:
            raise Exception("You need to fit the encoder first!")  
        return np.array([self.mapping[val] for val in X])  # Convert categories to numbers
    
    def fit_transform(self, X):
        self.fit(X)             # Fit mapping
        return self.transform(X)  # Transform categories
