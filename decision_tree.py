import math
import numpy as np

# Helper function to calculate entropy
def calculate_entropy(data):
    labels = [row[-1] for row in data]
    unique_labels = set(labels)
    entropy = 0

    for label in unique_labels:
        label_prob = labels.count(label) / len(labels)
        entropy += -label_prob * math.log2(label_prob)

    return entropy

# Split the dataset on a specific feature
def split_dataset(data, feature_index, threshold):
    left_split = [row for row in data if row[feature_index] <= threshold]
    right_split = [row for row in data if row[feature_index] > threshold]
    return left_split, right_split

# Calculate information gain
def calculate_info_gain(data, left_split, right_split):
    weight_left = len(left_split) / len(data)
    weight_right = len(right_split) / len(data)
    total_entropy = calculate_entropy(data)
    left_entropy = calculate_entropy(left_split)
    right_entropy = calculate_entropy(right_split)

    info_gain = total_entropy - (weight_left * left_entropy + weight_right * right_entropy)
    return info_gain

# Find the best split for a node
def find_best_split(data):
    best_gain = 0
    best_split = None
    best_feature = None
    n_features = len(data[0]) - 1  # Last column is the label
    
    for feature_index in range(n_features):
        values = set([row[feature_index] for row in data])
        for value in values:
            left_split, right_split = split_dataset(data, feature_index, value)
            if not left_split or not right_split:
                continue
            gain = calculate_info_gain(data, left_split, right_split)
            if gain > best_gain:
                best_gain, best_split, best_feature = gain, (left_split, right_split), (feature_index, value)
    
    return best_gain, best_split, best_feature

# Recursive function to build the tree
def build_tree(data, depth=0, max_depth=5):
    gain, split, feature = find_best_split(data)
    if gain == 0 or depth == max_depth:
        labels = [row[-1] for row in data]
        return max(set(labels), key=labels.count)  # Leaf node

    left_branch = build_tree(split[0], depth + 1, max_depth)
    right_branch = build_tree(split[1], depth + 1, max_depth)
    return {'feature_index': feature[0], 'threshold': feature[1], 'left': left_branch, 'right': right_branch}

# Make predictions
def predict(tree, row):
    if not isinstance(tree, dict):
        return tree
    if row[tree['feature_index']] <= tree['threshold']:
        return predict(tree['left'], row)
    else:
        return predict(tree['right'], row)

# Main function to demonstrate
def main():
    # Simple dataset: [Feature1, Feature2, Label]
    dataset = [
        [2.7, 2.5, 'A'],
        [1.3, 3.0, 'B'],
        [3.6, 1.8, 'A'],
        [2.0, 3.1, 'B'],
        [3.0, 2.0, 'A']
    ]
    
    # Build the decision tree
    tree = build_tree(dataset, max_depth=3)
    print(f"Decision Tree: {tree}")

    # Predict using the tree
    for row in dataset:
        prediction = predict(tree, row)
        print(f"Predicted: {prediction}, Actual: {row[-1]}")

if __name__ == "__main__":
    main()
