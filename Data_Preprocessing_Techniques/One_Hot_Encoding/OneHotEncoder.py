class OneHotEncoder:
    def __init__(self):
        self.categories_ = None  # Stores unique categories learned from data
    
    def fit(self, X):
        self.categories_ = np.unique(X)  # Learn unique categories
    
    def transform(self, X):
        if self.categories_ is None:
            raise Exception("You need to fit the encoder first!")  
        
        one_hot = np.zeros((len(X), len(self.categories_)))  # Initialize zero matrix
        category_to_index = {cat: idx for idx, cat in enumerate(self.categories_)}  # Map categories to columns
        
        for i, val in enumerate(X):
            one_hot[i, category_to_index[val]] = 1  # Set 1 at the correct column
        
        return one_hot
    
    def fit_transform(self, X):
        self.fit(X)                # Learn categories
        return self.transform(X)   # Transform to one-hot
