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
    
    def is_pure(self):
        # We're going to determine purity of our data based upon probabilities.
        p = self.probs()
        if p[0] == 1 or p[1] == 1:
            return True
        return False
    

    def split(self, depth=0):
        # Make our split
        X, y = self.X, self.y
        # Check for leaf status and purity.
        if self.is_leaf and not self.is_pure():
            # Find our best splits based upon the features, target, and column for every column in our features.
            splits = [self.find_best_split(X, y, column) for column in range(X.shape[1]) ]
            splits.sort()  # Sort the list we made with list comprehension
            gini, split_point, column = splits[0]  # Define our gini impurity, split point, and column selected.
            self.is_leaf = False  # Since we're making a new split, we're gonna need to make sure it's not a leaf.
            self.column = column 
            self.split_point = split_point
            
            left = X[:,column] <= split_point  # Set it below if X split at ,column is less than or equal to our split point.
            right = X[:,column] > split_point   # Set it above if greater than.
            
            # We need to define new children nodes for these values.
            self.children = [
                Node(X[left], y[left]),
                Node(X[right], y[right])
            ]
            # If we still have depth available
            if depth:
                for child in self.children:
                    child.split(depth-1)  # Split, subtracting 1 to reduce our available depth.
    
    def find_best_split(self, X, y, column):
        # If we sort our target vector by the values of X,
        # counting classes to go above / below our split point will be easy.
        self.X = X
        self.y = y
        self.column = column
        #### ORIGINAL METHOD
        # order = np.argsort(X[:,column])
        # classes = y[order]

        # # How many of each class are present to the left of our candidate split point?
        # class_0_left = (classes == 0).cumsum()  # Cumsum computes sum of arrays over axis.
        # class_1_left = (classes == 1).cumsum()

        # # Subtracting the cumulative sum from our total will give us our reverse cumulative sum.
        # class_0_right = class_0_left[-1] - class_0_left
        # class_1_right = class_1_left[-1] - class_1_left

        # # Calculate the totals for left and right.
        # left_total = class_0_left + class_1_left
        # right_total = class_0_right + class_1_right

        # # Formula for Gini Impurity:
        # # (class_1_left / left_total)* (class_0_left / left_total)
        # gini = (class_1_left / left_total)* (class_0_left / left_total) + (class_1_right / right_total)* (class_0_right / right_total)
        # gini[np.isnan(gini)] = 1



        # # Reverse the sorting to make sure we follow the rule
        # # C < split_value
        # best_split_gini = gini[np.argmin(gini)]  # Define our best split based upon minimum gini
        # best_split_index = np.argwhere(order == np.argmin(gini)).item(0)  # Define our best split's index.
        # best_split = X[best_split_index, column]  # Store best split based upon the index and column.

        # return best_split_gini, best_split, column

        #### WORKING ON IMPROVING
        # n_instances = float(sum([len(group) for group in X]))
        # gini = 0.0
        # for group in X:
        #     size = float(len(group))
        #     # We need to avoid a divide by zero error.
        #     if size == 0:
        #         continue
        #     score = 0.0
        #     # Score our group based upon score for each class.
        #     for class_value in classes:
        #         # scr = [row[0] for row in group].count(class_value) / size
        #         # score += scr * scr    
        #     # Weight score by size
        #     gini += (1.0 - score) * (size / n_instances)
        #     best_split = X[np.argwhere(order == gini)]

        # return gini, best_split, column 

    def probs(self):
        # We're gonna calculate our probabilities using the mean.
        return np.array([
            np.mean(self.y == 0),
            np.mean(self.y == 1),
        ])

    def predict_prob(self, row):
        if self.is_leaf:
            return self.probs()  # If it's a leaf, send to probabilities.
        else:  # If it's not a leaf...
            if row[self.column] <= self.split_point:  # Check if our column <= our split point.
                return self.children[0].predict_prob(row)  # If it is, recursion on 0th index.
            else:  # If it isn't...
                return self.children[1].predict_prob(row)  # Recursion on 1st index.

class Classifier:
    def __init__(self, max_depth=3):
        self.max_depth = int(max_depth)
        self.root = None
        
    def fit(self, X, y):
        self.root = Node(X, y)
        self.root.split(self.max_depth)  # Split with the selected maximum depth.
        
    def predict_prob(self, X):
        # Store results in an array.
        results = []
        for row in X:  # For every row in our feature,
            p = self.root.predict_prob(row)  # Recursive methodology to predict proba
            results += [p]  # Add the proba to our results.
        return np.array(results)  # Return as an array.
            
    def predict(self, X):
        # Make a prediction based upon method to predict proba.
        return (self.predict_prob(X)[:,1] > 0.5).astype(int)