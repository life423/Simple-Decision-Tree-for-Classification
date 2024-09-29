# Simple Decision Tree from Scratch

This repository provides a basic yet complete implementation of a decision tree classifier from scratch in Python. The algorithm is designed for binary and multi-class classification problems and demonstrates fundamental concepts in decision trees such as entropy, information gain, and recursive tree construction. 

## Project Structure
- **`decision_tree.py`**: The main script that implements the decision tree logic, from calculating entropy to making predictions based on the constructed decision tree.
- **`README.md`**: This documentation file explaining the project, features, and usage instructions.

## Features
This implementation includes several key features commonly found in decision trees:

- **Entropy Calculation**: Uses the concept of entropy from information theory to measure the uncertainty in a dataset. This metric is used to guide the splitting of nodes in the tree.
  
- **Information Gain**: For each possible feature and threshold, the algorithm computes the information gain, which represents the reduction in entropy achieved by splitting the dataset on a particular feature. The feature with the highest information gain is chosen as the splitting criterion.

- **Dataset Splitting**: The dataset is recursively split into left and right branches based on feature values, forming a binary decision tree.

- **Tree Recursion and Leaf Nodes**: The algorithm recursively splits the dataset until either a stopping condition is met (such as a maximum tree depth) or no further information gain is possible. Leaf nodes represent the final classification decision.

- **Prediction**: Once the decision tree is built, it can be used to classify new data points by traversing the tree from the root to the appropriate leaf node based on the feature values of the data point.

## Example Dataset
The dataset used in the script is a small, synthetic dataset containing two features and a label. Here’s an example of how the dataset is structured:
```yaml
[
  [2.7, 2.5, 'A'],
  [1.3, 3.0, 'B'],
  [3.6, 1.8, 'A'],
  [2.0, 3.1, 'B'],
  [3.0, 2.0, 'A']
]

Each row contains two numerical features and a categorical label ('A' or 'B'). The decision tree will attempt to split the dataset based on these feature values to classify future data points.

## How to Run the Script
Ensure you have Python installed on your system. The script uses only standard Python libraries like `math` and `numpy`, which can be installed with:

```bash
pip install numpy
