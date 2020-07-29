# Decision-Tree-Algorithm-LSCS by Kenneth T. Barrett
Decision Tree Algorithm for Lambda School's Computer Science Unit 1 Build Week

## Methodology

### U - Understand the Problem
A Decision Tree calculates the best split, based upon a scoring metric, in order to make predictions from training data on test data.
We're going to need to have a .fit() method for the Decision Tree to learn from our data, as well as a .predict() method to predict what
our values might be based upon what the Decision Tree has learned. In order to do this, we're going to need to have a way to split our data into
data to train our model, and data to test it, so we can see how well it's generalizing.

### P - Planning Phase
I'm going to use two separate classes - a node class, as well as a class for our Decision Tree. I've started with just a regression model to test out my methodology, and plan to move on to a classifier for discrete, categorical values. The class for the actual Decision Tree will simply have two methods: .fit() and .predict(). This being said, the node class will contain a variety of helper methods. Of these methods, we're going to need the following:

- Constructor
- A way to calculate all splits.
- A way to find the best possible split.
- A way to calculate the score in order to figure out said best split.
- A way to split feature columns.
- A way to determine if a value is a leaf.
- A way to make predictions based upon rows of features.

### E - Execution
Progress notes:
- I have the constructor complete, outside of the scoring instantiation.
- Finding the splits is mostly done; TODO: Set the nodes for left and right.
- Finding the best score is getting there; TODO: Set the left/right nodes, and change the way scoring is updated per chosen metric.
- I'm relatively sure the way I'm splitting columns won't work. TODO
- TODO: Determine if a value is a leaf.
- I'm not sure if our make_prediction function works, but it's getting there!

### R - Reflect