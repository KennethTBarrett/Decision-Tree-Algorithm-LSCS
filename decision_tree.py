import numpy as np

class DecisionTreeRegressor:
    """Decision Tree Regression model.
    Methods: `.fit()` and `.predict()`"""
    def fit(self, X, y, min_leaf=5):
        """Fits Decision Tree Regressor to our training data.
        Usage: `regressor.fit(X_train, y_train)`"""
        # Defining our tree, with our selected indices as a range
        # of our `y` value's length.
        self.tree = DecisionPoint(X, y, np.array(np.arange(len(y))), min_leaf)
        return self
    
    def predict(self, X):
        return self.tree.predict(X.values)  # Make prediction based upon our features.

class DecisionPoint:
    """Decision Tree Node class."""
    def __init__(self, X, y, indices, min_leaf=5):
        self.X = X  # Features
        self.y = y  # Target
        self.indices = indices  # Selected indices
        self.min_leaf = min_leaf  # Minimum number of leaves.
        self.num_rows = len(indices)  # Number of rows in our dataset.
        self.num_cols = X.shape[1]  # Will get the number of columns in features.
        self.value = np.mean(y)  # Mean value of our target at specified indices.
        self.score # TODO: SCORING
        self.find_split()  # Go ahead, start finding the initial split.


    def find_split(self):
        """Finds a split, then passes on to attempt to find better available split."""
        for col in range(self.num_cols):
            # For every column we have, let's find the best split.
            self.find_best_split(col)
        if self.is_leaf:  # If our point is a leaf...
            return None # Return NoneType
        X = self.split_column()  # Split our column using split_column helper function to use as X.
        # TODO: Set the left and right values.

    def find_best_split(self, var_idx):
        """Finds a better split, to ensure we have the best split."""
        # Use the indices selected in X values, as well as our
        # variable index.
        X = self.X.values[self.indices, var_idx]

        # For every row that we have...
        for row in range(self.num_rows):
            # TODO: Determine our left and right values.
            # TODO: Ensure we should be continuing.
            # Let's calculate our score based upon our metric (RMSE for testing).
            current_score = self.calculate_score(left, right)
            # If the score is less than the metric (RMSE)...
            if current_score < self.score:
                # Redefine our variable index, score, and the split to use.
                self.var_idx = var_idx
                self.score = current_score
                self.split = X[row]

    def split_column(self):
        """Splits our columns"""
        # Returns the split columns by indices and our variable index.
        return self.X.values[self.indices, self.var_idx]
    
    def is_leaf(self):
        """Determines if our value is our leaf."""
        # TODO: Determine if the value is our leaf.
        pass

    def calculate_score(self, left, right):
        #TODO: RMSE to start, then other metrics.
        pass

    def make_prediction(self, X):
        """Makes prediction for target based upon features."""
        # Check if it's a leaf. If it is...
        if self.is_leaf:
            return self.value  # Go ahead and return it.
        # We need to make our decision point the left, and x at the variable index.
        # if it's less than or equal to the split, otherwise, we need to go
        # to the right.
        if X[self.var_idx] <= self.split:
            node = self.left 
        else:
            node = self.right
        # Recursion.
        return node.make_prediction(X)
