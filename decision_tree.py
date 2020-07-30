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
        # We're going to determine purity of our data based upon .probs().
        # If there's a 100% distribution, the data is pure.
        # If there's not, the data is impure.
        p = self.probs()
        if p[0] == 1 or p[1] == 1:
            return True
        return False

    def split(self, depth=0):
        """Make splits, determining the left and right child nodes, until
        maximum depth has been reached, or no more splits can be made."""
        # Make our split
        X, y = self.X, self.y
        # Check for leaf status and purity.
        if self.is_leaf and self.is_data_pure() is False:
            # Find the best splits to make and store in list.
            splits = ([self.find_best_split(X, y, column)
                       for column in range(X.shape[1])])
            # Sort the list we made with list comprehension.
            splits.sort()

            # Fetch information returned from find_best_split()
            best_split_gini, split_point, column = splits[0]

            # Because new split is being made, not a leaf.
            self.is_leaf = False

            # Define column and split_point
            self.column = column
            self.split_point = split_point

            # Left if less than or equal to split point.
            # Right if greater than split point.
            left = X[:, column] <= split_point
            right = X[:, column] > split_point

            # We need to define new children nodes for these values.
            self.children = [
                Node(X[left], y[left]),
                Node(X[right], y[right])
            ]
            # If we still have depth available
            if depth:
                for child in self.children:
                    # Make split for each new child node,
                    # and subtract 1 from the available depth.
                    child.split(depth-1)

    def find_best_split(self, X, y, column):
        """Finds best available split based upon Gini"""
        # If we sort our target vector by the values of X,
        # counting classes to go left / right of our split point will be easy.
        self.X = X
        self.y = y
        self.column = column

        # Order our values for ease.
        order = np.argsort(X[:, column])
        classes = y[order]

        # How many of each class are present to the left of split point?
        # NOTE: .cumsum() computes cumulative sum of arrays over axis.
        class_0_left = (classes == 0).cumsum()
        class_1_left = (classes == 1).cumsum()

        # Subtracting the cumulative sum from our total will
        # give us our reverse cumulative sum.
        class_0_right = class_0_left[-1] - class_0_left
        class_1_right = class_1_left[-1] - class_1_left

        # Calculate the totals for left and right.
        left_total = class_0_left + class_1_left
        right_total = class_0_right + class_1_right

        # Ignore Divide by Zero Warning (Let it just Continue)
        np.seterr(divide='ignore', invalid='ignore')

        # Calculate Gini
        gini = ((class_1_left / left_total) * (class_0_left / left_total) +
                (class_1_right / right_total) * (class_0_right / right_total))

        # Deal with NaN values.
        gini[np.isnan(gini)] = 1

        # Minimum Gini is our best split
        best_split_rank = np.argmin(gini)
        # Define the best split in gini.
        best_split_gini = gini[best_split_rank]
        # The index of our best split; reverse sorting done above.
        best_split_index = np.argwhere(order == best_split_rank).item(0)
        # Use the information to define our best split.
        best_split = X[best_split_index, column]
        return best_split_gini, best_split, column

    def predict_prob(self, row):
        if self.is_leaf:
            return self.probs()  # If it's a leaf, send to probabilities.
        else:  # If it's not a leaf...
            if row[self.column] <= self.split_point:  # If <= split point...
                return self.children[0].predict_prob(row)  # Left child.
            else:  # If it isn't (> split point)...
                return self.children[1].predict_prob(row)  # Right child.

    def probs(self):
        # We're gonna determine our probabilities using the mean.
        return np.array([
            np.mean(self.y == 0),
            np.mean(self.y == 1),
        ])


class Classifier:
    def __init__(self, max_depth=5):
        self.max_depth = int(max_depth)
        self.root = None

    def fit(self, X, y):
        """Usage: clf.fit(X_train, y_train)
        Fit model to training data; learns relationship
        between features X and target y"""
        self.root = Node(X, y)  # Define our tree's root node.
        self.root.split(self.max_depth)  # Begin splitting process.

    def predict(self, X):
        """Usage: clf.fit(X_test)
        Make predictions on testing features, based upon what the model
        has learned about the data during fitting."""
        results = []
        for row in X:  # For every row in our feature,
            p = self.root.predict_prob(row)  # Predict probabilities.
            results += [p]  # Add to our results.
        return (np.array(results)[:, 1] > 0.5).astype(int)  # Prediction.
