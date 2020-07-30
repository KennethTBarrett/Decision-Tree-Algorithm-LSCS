import numpy as np

class Node:
    def __init__(self, X, y):
        # X is our features
        # y is our target
        self.X = X
        self.y = y
        self.is_leaf = True
        self.column = None
        self.split_point = None
        self.children = None
    
    def is_data_pure(self):
        # We're going to determine purity of our data based upon probabilities.
        # If there's a 100% distribution, the data is pure. If there's not, it's
        p = self.probs()
        if p[0] == 1 or p[1] == 1:
            return True
        return False
    

    def split(self, depth=0):
        """Makes splits for each column in our features, creates child nodes; recursive until max_depth has been reached."""
        # Make our split
        X, y = self.X, self.y
        # Check for leaf status and purity.
        if self.is_leaf and self.is_data_pure():
            # Find our best splits based upon the features, target, and column for every column in our features.
            splits = [self.find_best_split(X, y, column) for column in range(X.shape[1]) ]
            splits.sort()  # Sort the list we made with list comprehension

            gini, split_point, column = splits[0] # Define our gini impurity, split point, and column selected.
            self.is_leaf = False  # Since we're making a new split, we're gonna need to make sure it's not a leaf.

            # Define column and split_point
            self.column = column 
            self.split_point = split_point
            
            left = X[:,column] <= split_point  # Left if X split at ,column is less than or equal to our split point.
            right = X[:,column] > split_point   # Right if greater than split point.
            
            # We need to define new children nodes for these values.
            self.children = [
                Node(X[left], y[left]),
                Node(X[right], y[right])
            ]
            # If we still have depth available
            if depth:
                for child in self.children:
                    child.split(depth-1)  # Recursively split, subtracting 1 to reduce our available depth.
    
    def find_best_split(self, X, y, column):
        """Finds best available split based upon Gini"""
        # If we sort our target vector by the values of X,
        # counting classes to go left / right of our split point will be easy.
        self.X = X
        self.y = y
        self.column = column

        # Order our values for ease.
        order = np.argsort(X[:,column])
        classes = y[order]

        # How many of each class are present to the left of our candidate split point?
        class_0_left = (classes == 0).cumsum()  # Cumsum computes cumulative sum of arrays over axis.
        class_1_left = (classes == 1).cumsum()

        # Subtracting the cumulative sum from our total will give us our reverse cumulative sum.
        class_0_right = class_0_left[-1] - class_0_left
        class_1_right = class_1_left[-1] - class_1_left

        # Calculate the totals for left and right.
        left_total = class_0_left + class_1_left
        right_total = class_0_right + class_1_right

        np.seterr(divide='ignore', invalid='ignore')  # Ignore Divide by Zero Warning (Let it just Continue)

        # Calculate Gini
        gini = (class_1_left / left_total) * (class_0_left / left_total) + (class_1_right / right_total) * (class_0_right / right_total)
        gini[np.isnan(gini)] = 1

        # Reverse the sorting to make sure we follow the rule
        # C < split_value
        best_split_gini = gini[np.argmin(gini)]  # Define our best split based upon minimum gini
        best_split_index = np.argwhere(order == np.argmin(gini)).item(0)  # Define our best split's index.
        best_split = X[best_split_index, column]  # Store best split based upon the index and column.

        return best_split_gini, best_split, column


    def probs(self):
        # We're gonna determine our probabilities using the mean.
        return np.array([
            np.mean(self.y == 0),
            np.mean(self.y == 1),
        ])

    def predict_prob(self, row):
        if self.is_leaf:
            return self.probs()  # If it's a leaf, send to probabilities.
        else:  # If it's not a leaf...
            if row[self.column] <= self.split_point:  # Check if our column <= our split point.
                return self.children[0].predict_prob(row)  # If it is, left child.
            else:  # If it isn't...
                return self.children[1].predict_prob(row)  # Right child.

class Classifier:
    def __init__(self, max_depth=3):
        self.max_depth = int(max_depth)
        self.root = None
        
    def fit(self, X, y):
        """Usage: clf.fit(X_train, y_train)
        Fit model to training data; learn relationship between features X and target y"""
        self.root = Node(X, y)  # Define our root node from which to begin the splitting process.
        self.root.split(self.max_depth)  # Split with the selected maximum depth.

    def predict(self, X):
        """Usage: clf.fit(X_test)
        Make predictions on testing features, based upon what the model has learned about
        the data from fitting."""
        results = []
        for row in X:  # For every row in our feature,
            p = self.root.predict_prob(row)  # Recursive methodology to predict proba
            results += [p]  # Add the proba to our results.
        results = np.array(results)
        return (results[:,1] > 0.5).astype(int)  # Return predictions.