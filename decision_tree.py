import numpy as np
from collections import Counter


class Node:
    def __init__(self, left=None, right=None, best_feature=None, threshold=None, value=None):
        self.left = left
        self.right = right
        self.best_feature = best_feature
        self.threshold = threshold
        self.value = value
    
    def is_node_leaf(self):
        """Determines if node is a leaf node."""
        return self.value is not None

    
class Classifier:
    def __init__(self, max_depth=5):
        self.root = None
        self.max_depth = max_depth

    def fit(self, X, y):
        self.root = self._make_split(X, y)
        return self
    
    def predict(self, X):
        """Makes prediction by traversing down our decision tree."""
        return np.array([self._make_prediction(feature, self.root) for feature in X])

    def _make_split(self, X, y, depth=0):
        # Set our number of samples and features according to feature input shape.
        num_features = X.shape[1]
        num_labels = len(np.unique(y))  # How many unique labels?
        # Check if base case is met:
        # Maximum depth has been reached or only one label
        if (depth >= self.max_depth
                or num_labels == 1):
            leaf_value = self._most_common(y)
            return Node(value=leaf_value)  # Create leaf node if so.
        
        # Select the indices of a few random features to be used in information gain.
        random_features = np.random.choice(num_features, num_features, replace=False)

        # Use greedy search to find our best available split, according to information gain calculations.
        best_feature, threshold = self._find_best(X, y, random_features)

        # Determine left and right indices (left <= threshold, and right > threshold).
        left_idx, right_idx = self._what_splits(X[: ,best_feature], threshold)
        
        # Recursively make splits until base case is met, adding 1 to the depth each time.
        left = self._make_split(X[left_idx, :], y[left_idx], depth+1)
        right = self._make_split(X[right_idx, :], y[right_idx], depth+1)

        return Node(left, right, best_feature, threshold)

    def _find_best(self, X, y, feature_indices):
        # Define default values for best gain, index, and threshold.
        best_gain = -1
        split_index = None
        split_threshold = None
        # Iterate through list of feature indices.
        for index in feature_indices:
            # Set the selected column
            X_column = X[:, index]
            thresholds = np.unique(X_column)  # Determine unique values.
            for threshold in thresholds:
                # Iterate through and calculate the information gain.
                gain = self._calculate_information_gain(X_column, y, threshold)
                
                # If the gain we calculated is better than the previous best gain
                # we're going to use it to set our split index and threshold, as
                # well as re-define the current best gain for our next pass.
                if gain > best_gain:
                    best_gain = gain
                    split_index = index
                    split_threshold = threshold
        return split_index, split_threshold

    def _what_splits(self, X, split_threshold):
        # Split should be left if X <= our threshold.
        left = np.argwhere(X <= split_threshold).flatten()
        # Split should be right if X > our threshold.
        right = np.argwhere(X > split_threshold).flatten()
        return left, right

    def _calculate_entropy(self, y):
        histo = np.bincount(y)  # Counts occurence of each element
        ps = histo / len(y)
        # List comprehension; calculate entropy.
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    
    def _calculate_information_gain(self, X, y, split_threshold):
        # Calculate entropy of the parent
        parent_entropy = self._calculate_entropy(y)
        # Determine the left and right values.
        left, right = self._what_splits(X, split_threshold)

        # Confirm there's information gain.
        if len(left) == 0 or len(right) == 0:
            return 0

        # Calculate the weighted average of child entropy
        num = len(y)
        num_left, num_right = len(left), len(right)
        # Calculate the entropy for the left side, as well as the right.
        entropy_left, entropy_right = self._calculate_entropy(y[left]), self._calculate_entropy(y[right])

        # Determine the entropy of the children, using the formula:
        # (length of the values on the left / total number of values) * left side entropy +
        # (length of the values on the right / total number of values) * right side entropy
        child_entropy = (num_left/num) * entropy_left + (num_right/num) * entropy_right

        # Return information gain by subtracting the child entropy from the parent entropy.
        information_gain = parent_entropy - child_entropy
        return information_gain


    def _most_common(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def _make_prediction(self, X, node):
        # Check if the node is a leaf. If it is, return its value.
        if node.is_node_leaf():
            return node.value
        # Otherwise, traverse down the decision tree.
        if X[node.best_feature] <= node.threshold:
            return self._make_prediction(X, node.left)
        else:
            return self._make_prediction(X, node.right)