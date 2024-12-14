import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

class KNNScratch:
    def __init__(self, neighbors=5, metric='euclidean', p=5):
        self.neighbors = neighbors
        self.metric = metric
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def distance(self, x1, x2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum(np.square(x1 - x2))) 
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2)) 
        elif self.metric == 'minkowski':
            return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)
        
    def predict(self, X_test):
        predictions = []

        for x_test in X_test:
            distances = [self.distance(x_test, x_train) for x_train in self.X_train]
            sorted_distances_indexes = np.argsort(distances)[:self.neighbors]
            k_nearest = [self.y_train[i] for i in sorted_distances_indexes]
            most_class = Counter(k_nearest).most_common(1)
            predictions.append(most_class[0][0])
        
        return predictions

class GaussianNaiveBayesScratch:
    def gauss_dist(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = np.zeros((len(self.classes), X.shape[1]))
        self.var = np.zeros((len(self.classes), X.shape[1]))
        self.priors = np.zeros(len(self.classes))

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        likelihoods = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            likelihood = np.sum(np.log(self.gauss_dist(idx, x)))
            likelihood += prior
            likelihoods.append(likelihood)
        return self.classes[np.argmax(likelihoods)]

import numpy as np

class DecisionTreeClassifierScratch:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
    
    def entropy(self, y):
        """Calculate the entropy of a dataset."""
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Added epsilon to avoid log(0)

    def information_gain(self, y, y_left, y_right):
        """Calculate information gain from a split."""
        parent_entropy = self.entropy(y)
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        if n_left == 0 or n_right == 0:
            return 0
        # Weighted average of the entropy of the children
        child_entropy = (n_left / n) * self.entropy(y_left) + (n_right / n) * self.entropy(y_right)
        return parent_entropy - child_entropy

    def split(self, X, y, feature_index, threshold):
        """Split the dataset into left and right based on a threshold."""
        left_idxs = X[:, feature_index] <= threshold
        right_idxs = X[:, feature_index] > threshold
        return X[left_idxs], X[right_idxs], y[left_idxs], y[right_idxs]

    def best_split(self, X, y):
        """Find the best feature and threshold to split on using information gain."""
        best_gain = -1
        best_index, best_threshold = None, None

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self.split(X, y, feature_index, threshold)
                
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gain = self.information_gain(y, y_left, y_right)

                if gain > best_gain:
                    best_gain = gain
                    best_index = feature_index
                    best_threshold = threshold

        return best_index, best_threshold

    def build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if n_labels == 1 or n_samples == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return np.bincount(y).argmax()

        feature_index, threshold = self.best_split(X, y)
        if feature_index is None:
            return np.bincount(y).argmax()

        left_idxs = X[:, feature_index] <= threshold
        right_idxs = X[:, feature_index] > threshold
        left = self.build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self.build_tree(X[right_idxs], y[right_idxs], depth + 1)

        return {'feature_index': feature_index, 'threshold': threshold, 'left': left, 'right': right}

    def fit(self, X, y):
        """Build the decision tree from the training data."""
        self.tree = self.build_tree(X, y)

    def predict_one(self, x, tree):
        """Predict a single instance using the decision tree."""
        if not isinstance(tree, dict):
            return tree
        if x[tree['feature_index']] <= tree['threshold']:
            return self.predict_one(x, tree['left'])
        else:
            return self.predict_one(x, tree['right'])

    def predict(self, X):
        """Predict multiple instances using the decision tree."""
        return np.array([self.predict_one(x, self.tree) for x in X])

iris = datasets.load_iris()
X = iris.data 
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# knn = KNNScratch(neighbors=3, metric='euclidean')
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train, y_train)
# predictions = knn.predict(X_test)

# gnb = GaussianNaiveBayesScratch()
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)
# predictions = gnb.predict(X_test)

ct = DecisionTreeClassifierScratch()
# ct = DecisionTreeClassifier()
ct.fit(X_train, y_train)
predictions = ct.predict(X_test)

sc = DecisionTreeClassifier()
sc.fit(X_train, y_train)
predictionssc = sc.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
print("Accuracy:", accuracy)
print("Classification Report:\n", report)


print("Accuracy:", accuracy_score(y_test, predictionssc))
print("Classification Report:\n", classification_report(y_test, predictionssc))