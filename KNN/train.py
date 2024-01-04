import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN import KNN

# Color map for visualizing the classes
CMAP = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Loading the Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Plotting the dataset
# Here, only two features are used for visualization (feature 3 vs feature 4)
plt.figure()
plt.scatter(X[:, 2], X[:, 3], c=y, cmap=CMAP, edgecolors='k', s=20)
plt.show()

# Creating an instance of the KNN classifier and fitting it to the training data
clf = KNN(k=5)
clf.fit(X_train, y_train)

# Making predictions on the test set
predictions = clf.predict(X_test)

# Calculating the accuracy of the model
acc = np.sum(predictions == y_test) / len(y_test)
print(acc)  # Note: Variable names are case-sensitive in Python, so use 'acc', not 'ACC'
