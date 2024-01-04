import numpy as np
from collections import Counter


# Function to calculate Euclidean distance between two points
def euclidean_distance(x1, x2):
    # Sum of squared differences
    # Square root of the sum provides the Euclidean distance
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance


# KNN Classifier
class KNN:
    def __init__(self, k=3):
        # Constructor to initialize the number of neighbors (k)
        self.k = k

    def fit(self, X, y):
        # Method to train the classifier
        # X_train: training data features
        # y_train: training data labels
        self.X_train = X
        self.y_train = y

    def predict(self, X_pred):
        # Method to predict the class for new data points
        # X_pred: new data points for which the labels are to be predicted
        # Returns a list of predicted labels
        predictions = [self._predict(x) for x in X_pred]
        return predictions

    def _predict(self, x):
        # Private method to predict the label for a single data point
        # x: a single data point

        # Calculate the Euclidean distance of 'x' from all points in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Sort distances and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Extract the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Determine the most common class label in the k nearest neighbors
        # Counter.most_common returns a list of tuples (label, count)
        most_common = Counter(k_nearest_labels).most_common()

        # Return the most common class label
        return most_common[0][0]
